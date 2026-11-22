# Traces

A trace is a coding-agent session, recorded next to the runs it produced. The
watcher uploads it to the manager, where the dashboard's **Traces** tab shows
it next to the notes: the conversation that built the experiment, searchable
months later.

## What is captured

> Sent

* your prompts
* the assistant's replies
* thinking blocks
* tool calls, as one summary line: the shell command for bash-like tools,
  paths and scalar arguments elsewhere
* tool results as status only (ok / error / how long)

> Not sent

* tool output
* file contents
* the system prompt

Nothing is redacted and nothing is streamed: a session's new events are
uploaded in one batch once its transcript has been quiet for a minute, and
again when the session closes. A trace is a few kilobytes.

## Turning it on

On the machine the coding agents run on (the manager's own machine, usually):

```sh
mlsweep capture on
```

That watches pi and Claude Code transcripts and uploads new sessions as they
happen. It stays off until you run this; the manager also accepts nothing when
started with `--no-capture`.

```sh
mlsweep capture off          # stops the watcher, removes the credential
mlsweep capture status       # on or off, and what the watcher is doing
mlsweep capture ls           # traces on the manager, newest first
```

`off` writes a marker (who, when) that a running watcher checks every loop, so
"off" holds even if stopping the process failed. Traces already uploaded stay
on the manager; delete those from the Traces tab.

### Agents

Capture tails the transcripts the agents themselves write, so it needs no
plugin inside the agent and works the same for every one it knows:

| Agent | Transcripts |
| - | - |
| pi | `~/.pi/agent/sessions/**.jsonl` |
| Claude Code | `~/.claude/projects/**.jsonl` |

## Attribution

An experiment records the trace it was submitted from. The watcher exports
nothing; the client reads the session's id out of the environment the agent
runs its commands with:

* pi exports `PI_SESSION_ID` into every child process, so a `mlsweep run` the
  agent types carries its own session's id;
* `MLSWEEP_SESSION_ID` overrides everything, for agents that name no session;
* otherwise the client walks its `/proc` ancestors and matches them against
  the watcher's registry of running agent processes, then against the most
  recent session whose working directory matches an ancestor's.

Every answer is exact per agent context — two agents in one directory are two
sessions and never cross-link. A human running `mlsweep run` links nothing.

`mlsweep capture link EXP_ID` links the current session (or `--session ID`) to
an experiment by hand.

## Reading traces

The **Traces** tab lists the sessions every linked machine uploaded. A trace
page is a ledger: a narrow lane of event tags, then the content — your
prompts, the agent's replies, its thinking collapsed and dimmed, tool calls
one line each. Clicking an event opens the inspector with the full text and
its metadata (model, tokens). A search box over the sidebar searches
every trace's text; a second box filters the open trace's events.

An experiment page links to the traces that submitted it, the other way
around too.

## Notes

A note can link to a trace with the evidence-link syntax:

```markdown
The sweep came out of [that idea about warmup](mlsweep:trace/9f6c0e22).
```

`mlsweep notes lint` checks that the trace exists on the manager, like every
other `mlsweep:` link. (Yes, a trace shows the very session writing the link
that points at itself.)

## Importing history instead

To bring in the transcripts you already have:

```sh
mlsweep capture import --days 30     # --all for everything, --dry-run to look
```

Each imported file becomes a trace marked `imported`, and is never imported
twice.
