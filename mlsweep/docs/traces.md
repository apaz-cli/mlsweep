# Traces

A trace is a coding-agent session, recorded next to the runs it produced. The
watcher uploads it to the manager, where the dashboard's Traces tab shows
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

Nothing is redacted and nothing is streamed. A followed session uploads in one
batch a few seconds after it is followed, then again whenever its transcript has
been quiet for a minute, and once more when the session closes. A trace is a
few kilobytes.

## Which sessions

A session is uploaded when it asks, or when it submits a sweep while capture is
on. Every other transcript on the machine stays unread. To ask, run this inside
the agent session you want on the Traces tab (or tell the agent to run it):

```sh
mlsweep capture this
```

That session uploads from its first prompt, and its new turns follow. If capture
was never set up on this machine, `capture this` turns it on.

With capture on, a session also uploads once it submits or links an
experiment, so a `mlsweep run` typed by the agent brings its trace along:

```sh
mlsweep capture on           # sweep-submitting sessions upload too; reads nothing yet
mlsweep capture off          # stops the watcher, removes the credential
mlsweep capture status       # on or off, and the sessions it follows
mlsweep capture ls           # traces on the manager, newest first
```

`off` writes a marker (who, when) that a running watcher checks every loop, so
"off" holds even if stopping the process failed. After `off`, `capture this`
refuses until someone runs `capture on` again. Traces already uploaded stay on
the manager; delete those from the Traces tab. The manager accepts nothing when
started with `--no-capture`.

### Agents

Capture reads the transcripts the agents themselves write, so it needs no
plugin inside the agent and works the same for every one it knows:

| Agent | Transcripts | Session variable |
| - | - | - |
| pi | `~/.pi/agent/sessions/**.jsonl` | `PI_SESSION_ID` |
| Claude Code | `~/.claude/projects/**.jsonl` | `CLAUDE_CODE_SESSION_ID` |

## Attribution

An experiment records the trace it was submitted from. The client reads the
session's id out of the environment the agent runs its commands with. Both
agents export their session variable into every child process, so a
`mlsweep run` the agent types carries its own session's id. When a command's
environment was scrubbed, the client reads its `/proc` ancestors' instead.
`MLSWEEP_SESSION_ID` overrides everything, for agents that name no session.

Nothing is guessed from working directories or recent files. Two agents in one
directory are two sessions and never cross-link, and a human running
`mlsweep run` links nothing.

`mlsweep capture link EXP_ID` links the current session (or `--session ID`) to
an experiment by hand, and follows it while capture is on.

## Reading traces

The Traces tab lists the sessions every linked machine uploaded, newest
activity first. The sidebar filters by agent and by state (live, closed,
imported), and its search box searches every trace's text, title and working
directory; a second box filters the open trace's events. A trace too far down
the list is reached with "Show more traces".

A trace page is a ledger: a time column, the event tag, then the content —
your prompts, the agent's replies, its thinking collapsed and dimmed, tool
calls one line each with their result's status and duration. Clicking an event
opens the inspector with the full text and its metadata (model, tokens), and a
copy button. The turn counter in the header steps between prompts (`[` and
`]`); `j`/`k` or the arrow keys move through events, `Enter` opens one, `Esc`
closes the inspector, and `/` jumps to the search box. The page's URL names the
open trace and the selected event, so a copied link comes back to them.

A trace records the experiments it submitted, and each chip links to one. `+`
links a trace the auto-link missed (choose the experiment from a list), and the
chip's `×` unlinks it. The `⋮` menu copies the trace id or its
`mlsweep:trace/<id>` reference for a note, downloads the transcript, or deletes
the trace. When notes are served, the header also says how many notes cite the
trace.

An experiment page links to the traces that submitted it, the other way
around too (its Traces entry filters this list to that experiment).

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
