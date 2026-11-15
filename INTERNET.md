# Exposing mlsweep to the internet

This is not safe yet. Keep the manager on a private network and reach it through
an SSH tunnel, Tailscale or WireGuard until the four decisions below are made and
carried out.

The fixes that change nothing for users are done and listed at the end. What
remains either changes how people use mlsweep or is new design work.

Findings marked *tested* were reproduced against a throwaway manager on
aiohttp 3.13.5 and OpenSSH 10.0. Findings marked *from code* come from reading the
source and were not run. Line numbers will drift.

## The core problem

The manager still has one token, and that token grants everything. Its holder can
run any command on every worker, since running a job means executing its `command`
and `setup_command`. The holder can also read all results, delete all data, edit
the notes and point the manager at new SSH hosts.

So the question for each item below is how the token could leak, and what an
attacker gets once it has. Today the answer to the second part is shell access on
the GPU machines.

## Decisions still open

### 1. Bind to loopback by default

`manager.py:314` hardcodes `0.0.0.0`. A `--bind` flag defaulting to `127.0.0.1`
makes exposure a choice. Anyone who opens the dashboard from another machine would
then have to pass `--bind`. Remote workers fetch artifacts through an SSH reverse
tunnel (*from code*), so they should keep working. The weaker fallback is to keep
the current default and print a warning at startup.

### 2. TLS

The manager speaks plain HTTP, so the token and any W&B API key in an export
request cross the wire in cleartext (*from code*). A reverse proxy should be the
documented deployment, and Caddy is the shortest path. Three things come with it:

- `http://` is hardcoded in `manager.py:300` and `manager.py:186`.
- The `https` paths through `mlsweep_run` and `ctl` have not been tested, and need
  a test that runs them behind a proxy.
- HSTS can be added once TLS exists.

### 3. A separate worker credential, kept out of `ps`

The manager launches workers with `--token TOKEN` (`_manager_workers.py:526`), and
it showed up in `ps` (*tested*). On a shared GPU machine, every local user can read
it. A training job runs as the same user as its worker, so it can also read the
token from the parent's `/proc` entry (*from code*). One bad dependency in training
code then controls every other machine.

The fix has two parts. Pass the secret over stdin or a 0600 file, and issue each
worker its own credential that works only on the worker protocol and artifact
downloads. Manager-launched workers would pick this up by themselves. People who
start workers by hand would change how they pass it.

### 4. The token in URLs and `localStorage`

`?token=` shows up in proxy logs, browser history and shell history (*from code*).
The sources today:

- the dashboard URL the manager prints (`manager.py:186`)
- `authURL` for downloads and images (`chrome.js:25`)
- the WebSocket URLs in `page-logs.js` and `page-results.js`
- artifact URLs sent to workers (`_manager_workers.py:1640`)

`chrome.js:11` also keeps the token in `localStorage`, so any XSS that gets past the
CSP becomes full compromise. `page-experiments.js` shows it in a copyable command.

Replace these with four things:

- A one-time code in the printed URL, exchanged for an `HttpOnly; Secure;
  SameSite=Strict` cookie. People would still click a link.
- Short-lived signed URLs for downloads.
- A header for worker fetches.
- An env var or file for `mlsweep_run` and `ctl`, with `--token` deprecated for a
  release.

Cookies also bring in CSRF tokens and an `Origin` check on the WebSocket. Scripts
that use `?token=` would move to the header.

## Terminals (`--no-terminals`)

Terminals are on by default: the manager serves shells on its own machine to the Notes tab,
in a bubblewrap sandbox (`_sessions.py`). `--no-terminals` turns them off. It adds no power the
token lacked, since the token already runs code on every worker. It does make an XSS
that gets past the CSP an interactive shell on the manager, which raises the stakes
of decision 4 above. The socket refuses other origins, and keystrokes cross the wire
in cleartext like everything else until decision 2 is made. The sandbox's shell holds
the manager token itself; scoped credentials would let it hold less.

## Smaller items that change behavior

Each of these is small, and each was left out because someone could notice it.

- Workers with no token accept anyone. `worker.py:332` skips the check when `--token`
  is empty, and `worker.py:1303` binds all interfaces. Refuse to start without a
  token unless bound to loopback. Hand-started workers without one would stop
  working.
- The worker protocol is plaintext TCP with the token in `MsgHello`. Have workers
  bind loopback and reach them through the SSH session the manager holds, or add TLS.
  Never expose this port.
- `--token` accepts any string and a token lasts forever. Add a minimum length and a
  rotate command.
- The notes page allows images from any host, so an image in a note can signal to an
  outside server when the note is read. Blocking that would break notes that embed
  remote images.
- `/api/notes/status` returns the absolute path and git remote. The Notes tab shows
  both.
- The manager calls ipify at startup, to print its public URL. It no longer sends
  the token anywhere, but it does tell a third party its address.

## Real access control

This is the redesign, and it is what several people on the internet would need.

### Scoped credentials

At minimum:

| Scope | Can do |
|---|---|
| read | dashboards, logs, metrics, downloads |
| notes | read and write notes |
| submit | create jobs, which is code execution on workers |
| admin | workers, scheduler, delete, purge, GC |
| worker | protocol and artifact fetches only |

### Secrets in the read scope

`GET /api/jobs` returned a job's `env` to the caller (*tested*). People put API keys
in there. Once a read scope exists it must hide `env`. `command` may need the same
treatment, since command lines often carry keys too.

### Campaigns as a boundary

`?campaign=` is an optional filter today. Omitting it removes the restriction
(`_manager_http.py:13`). Bind a credential to its campaigns if they are meant to
separate people.

### Notes access

The manager token is the only credential for the notes, over HTTP and over the git
remote at `/notes.git`. Its holder can read and rewrite any note, though the remote
refuses force pushes and branch deletes, so history stays recoverable through git.
`sources/` is a convention with no enforcement. A scope for notes would need to cover
both routes. Clones keep the token in `.git/mlsweep-token`, mode 0600.

### Server-side actions on hosts

`POST /api/workers` makes the manager SSH to any host using any key path on its
disk. Make it admin only, or restrict it to hosts listed in `workers.toml` when the
manager is exposed.

### Sandboxed workers

Anyone allowed to submit can run code as the worker's user, and no scope changes
that. Untrusted submitters need containers or unprivileged users, no credentials on
worker hosts, and restricted outbound network. This is the largest change for users,
since jobs that rely on the host's filesystem, environments or network would break.

### Backups

`DELETE`, purge and GC are permanent. Notes history survives in git, and nothing else
does.

## Checklist

A single trusted user behind a proxy needs the four decisions and the worker items:

- [ ] The default bind is loopback, or a warning is printed.
- [ ] The TLS proxy is documented and tested.
- [ ] The worker credential is separate, and no token appears in `ps`.
- [ ] No token appears in URLs or `localStorage`.
- [ ] Workers refuse to run without a token off loopback.
- [ ] Worker ports are firewalled or tunneled.
- [ ] The token has rotation and a minimum length.

Several people also need these:

- [ ] Credentials are scoped, with `env` hidden from read scope.
- [ ] Notes rules are enforced over HTTP.
- [ ] Credentials are bound to campaigns, if campaigns separate people.
- [ ] Worker management is limited to admin or a host allowlist.
- [ ] Backups exist.

Untrusted submitters also need these:

- [ ] Workers are sandboxed, with no credentials on the host.
- [ ] An outside security review has happened.

## Done

Tests are in `tests/test_security.py`. The byte and file-count limits, the zip limit,
the worker hello deadline, constant-time comparison and the `--` arguments have no
dedicated test.

Paths and IDs:

- Artifact, experiment and run IDs are checked on every route, so `%2F` tricks no
  longer reach the disk.
- Inserts refuse unsafe run and artifact IDs. Run names may still contain `/`, since
  they come from sweep option values.
- Purge and move stay inside their experiment for rows that already hold a bad ID.
- At startup the manager fails any unfinished job whose ID is unsafe.

Worker launches:

- A worker host that starts with `-` or holds whitespace is refused, in the API, in
  `workers.toml` and at launch.
- Every `ssh` and `scp` call puts `--` before the destination.

The token:

- `/api/reachable` is gone. The startup probe sends no token, and the dashboard reads
  the public URL from `/api/health`.
- Token comparison is constant time, on the API and on workers.
- Repeated bad tokens block an address for a minute. Loopback is exempt, which also
  exempts clients behind a local proxy.

The dashboard:

- Every page has a server-sent CSP with `script-src 'self'`. All inline scripts and
  handlers moved to files, so injected script does not run. The tests check this in
  headless Chromium.
- Files a job wrote can no longer run script on the dashboard's origin. HTML downloads
  instead of rendering, and SVG is sandboxed.
- Job data is escaped wherever the pages build markup from it.
- Plotly ships in `webui/`, and the dashboard no longer calls ipify.
- CORS is gone, directory listings are off, and every response has `nosniff` and
  `no-referrer`.

Limits and records:

- Request bodies are capped at 128 MB, and artifact uploads at 512 MB. Uploads stream
  to disk rather than into memory.
- WebSocket messages are capped at 1 MB, notes uploads at 1000 files, and zip exports
  at four at a time, with the rest waiting.
- A worker's first frame is capped at 1 MB with a 30-second deadline.
- `<mlsweep-dir>/audit.log` records every change and refused login, without query
  strings.
- A 500 no longer returns exception text to the client.
- The `aiohttp` floor is 3.14.5, which had no published advisories when this was
  written, and `.github/workflows/audit.yml` runs `pip-audit` weekly.

These were checked and needed no change:

- Every API route rejects a request without the token, and path tricks do not get past
  the auth middleware (*tested*).
- `manager.token` is written with mode 0600 (*from code*).
- The notes pre-receive hook refused a direct edit to a protected file, and also a
  crafted merge of a new root commit that tampered with protected files (*tested*).
