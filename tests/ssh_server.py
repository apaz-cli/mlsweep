#!/usr/bin/env python3
"""Simple SSH server using paramiko for localhost testing.
Handles exec requests with exit status and SFTP subsystem.

Usage:
    python3 ssh_server.py --port <port> --host-key <path> --authorized-keys <path>
"""

import argparse
import socket
import sys
import os
import threading
import select
import subprocess
import paramiko

HOST = '127.0.0.1'
PORT = 2222
HOST_KEY_PATH = '/workspace/.ssh/ssh_host_rsa_key'
AUTHORIZED_KEYS_PATH = '/workspace/.ssh/authorized_keys'


class SFTPServerInterface(paramiko.SFTPServerInterface):
    """Simple SFTP interface that maps to the real filesystem."""

    def __init__(self, server, *args, **kwargs):
        super().__init__(server, *args, **kwargs)
        self.home = '/workspace'

    def list_folder(self, path):
        full = os.path.normpath(os.path.join('/', path))
        try:
            entries = []
            for fname in os.listdir(full):
                attr = paramiko.SFTPAttributes.from_stat(os.stat(os.path.join(full, fname)))
                attr.filename = fname
                entries.append(attr)
            return entries
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def stat(self, path):
        full = os.path.normpath(os.path.join('/', path))
        try:
            s = os.stat(full)
            return paramiko.SFTPAttributes.from_stat(s)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def lstat(self, path):
        return self.stat(path)

    def open(self, path, flags, attr):
        full = os.path.normpath(os.path.join('/', path))
        try:
            fd = os.open(full, flags)
            return SFTPHandle(self, fd)
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def mkdir(self, path, attr):
        full = os.path.normpath(os.path.join('/', path))
        try:
            os.mkdir(full)
            return paramiko.SFTP_OK
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def rmdir(self, path):
        full = os.path.normpath(os.path.join('/', path))
        try:
            os.rmdir(full)
            return paramiko.SFTP_OK
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def remove(self, path):
        full = os.path.normpath(os.path.join('/', path))
        try:
            os.remove(full)
            return paramiko.SFTP_OK
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def rename(self, oldpath, newpath):
        old_full = os.path.normpath(os.path.join('/', oldpath))
        new_full = os.path.normpath(os.path.join('/', newpath))
        try:
            os.rename(old_full, new_full)
            return paramiko.SFTP_OK
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def symlink(self, target_path, path):
        full = os.path.normpath(os.path.join('/', path))
        try:
            os.symlink(target_path, full)
            return paramiko.SFTP_OK
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)

    def readlink(self, path):
        full = os.path.normpath(os.path.join('/', path))
        try:
            target = os.readlink(full)
            return target
        except OSError as e:
            return paramiko.SFTPServer.convert_errno(e.errno)


class SFTPHandle(paramiko.SFTPHandle):
    """Handle for an open SFTP file."""

    def __init__(self, sftp_server, fd):
        super().__init__()
        self.sftp_server = sftp_server
        self.fd = fd

    def close(self):
        os.close(self.fd)
        return paramiko.SFTP_OK

    def read(self, offset, length):
        os.lseek(self.fd, offset, os.SEEK_SET)
        data = os.read(self.fd, length)
        return data

    def write(self, offset, data):
        os.lseek(self.fd, offset, os.SEEK_SET)
        written = os.write(self.fd, data)
        return paramiko.SFTP_OK


class SSHServer(paramiko.ServerInterface):
    """Simple server that accepts public key auth, handles exec and SFTP."""

    def __init__(self):
        self.event = threading.Event()
        self.command = None
        self.subsystem = None

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_publickey(self, username, key):
        """Check the public key against authorized_keys."""
        try:
            with open(AUTHORIZED_KEYS_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        blob = paramiko.PublicBlob.from_string(line)
                    except (ValueError, paramiko.SSHException):
                        continue
                    if key.asbytes() == blob.key_blob:
                        return paramiko.AUTH_SUCCESSFUL
        except Exception as e:
            print(f"Error reading authorized_keys: {e}", file=sys.stderr)
        return paramiko.AUTH_FAILED

    def check_auth_password(self, username, password):
        return paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "publickey"

    def check_channel_shell_request(self, channel):
        self.event.set()
        return True

    def check_channel_exec_request(self, channel, command):
        self.command = command
        self.event.set()
        return True

    def check_channel_subsystem_request(self, channel, name):
        if name == 'sftp':
            self.subsystem = 'sftp'
            self.event.set()
            return True
        return False

    def check_channel_pty_request(self, channel, term, width, height,
                                  pixelwidth, pixelheight, modes):
        return True


def generate_host_key():
    """Generate a host key if one doesn't exist."""
    if not os.path.exists(HOST_KEY_PATH):
        print(f"Generating host key: {HOST_KEY_PATH}")
        key = paramiko.RSAKey.generate(2048)
        key.write_private_key_file(HOST_KEY_PATH)
        with open(HOST_KEY_PATH + '.pub', 'w') as f:
            f.write(f"{key.get_name()} {key.get_base64()} localhost\n")
    os.chmod(HOST_KEY_PATH, 0o600)


def handle_client(conn, addr, host_key):
    """Handle a single client connection."""
    print(f"Connection from {addr}")
    transport = paramiko.Transport(conn)
    transport.add_server_key(host_key)
    transport.local_version = f"SSH-2.0-paramiko_{paramiko.__version__}"

    server = SSHServer()
    try:
        transport.start_server(server=server)
    except paramiko.SSHException as e:
        print(f"SSH negotiation failed: {e}")
        return

    chan = transport.accept(20)
    if chan is None:
        print("Channel timeout")
        return

    server.event.wait(10)
    if not server.event.is_set():
        print("No shell/exec/subsystem request")
        chan.close()
        transport.close()
        return

    if server.subsystem == 'sftp':
        print("Starting SFTP subsystem")
        sftp_server = paramiko.SFTPServer(chan, 'sftp', server, SFTPServerInterface)
        # SFTPServer runs in the calling thread until the channel closes
        sftp_server.run()
    elif server.command is not None:
        # Execute the command
        cmd_str = server.command.decode('utf-8', errors='replace') if isinstance(server.command, bytes) else server.command
        print(f"Executing: {cmd_str}")
        try:
            proc = subprocess.run(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            output = proc.stdout
            exit_status = proc.returncode
        except subprocess.TimeoutExpired:
            output = b"Command timed out"
            exit_status = -1
        except Exception as e:
            output = f"Error: {e}".encode()
            exit_status = -1

        chan.send(output)
        chan.send_exit_status(exit_status)
        # Give client time to receive exit status before closing
        import time
        time.sleep(0.1)
    else:
        # Shell mode - just echo
        try:
            chan.send(b"Welcome to paramiko SSH server!\r\n")
            while True:
                r, w, x = select.select([chan], [], [], 1.0)
                if r:
                    data = chan.recv(1024)
                    if not data:
                        break
                    decoded = data.decode('utf-8', errors='replace')
                    if decoded.strip() == 'exit':
                        chan.send(b"Goodbye!\r\n")
                        break
                    chan.send(f"You typed: {decoded}".encode('utf-8', errors='replace'))
                if not transport.is_active():
                    break
            chan.send_exit_status(0)
        except Exception as e:
            print(f"Shell error: {e}")

    chan.close()
    transport.close()
    print(f"Connection from {addr} closed")


def main():
    global PORT, HOST_KEY_PATH, AUTHORIZED_KEYS_PATH

    parser = argparse.ArgumentParser(description='Simple paramiko SSH server for testing')
    parser.add_argument('--port', type=int, default=PORT, help=f'Port to listen on (default: {PORT})')
    parser.add_argument('--host-key', default=HOST_KEY_PATH, help=f'Path to SSH host key (default: {HOST_KEY_PATH})')
    parser.add_argument('--authorized-keys', default=AUTHORIZED_KEYS_PATH, help=f'Path to authorized_keys file (default: {AUTHORIZED_KEYS_PATH})')
    args = parser.parse_args()

    PORT = args.port
    HOST_KEY_PATH = args.host_key
    AUTHORIZED_KEYS_PATH = args.authorized_keys

    generate_host_key()
    host_key = paramiko.RSAKey(filename=HOST_KEY_PATH)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(10)
    print(f"SSH server listening on {HOST}:{PORT}")

    while True:
        conn, addr = sock.accept()
        # Handle each connection in a thread (or fork)
        t = threading.Thread(target=handle_client, args=(conn, addr, host_key))
        t.daemon = True
        t.start()


if __name__ == '__main__':
    main()
