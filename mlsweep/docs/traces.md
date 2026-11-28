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

With capture on, a session also uploads once it submits an experiment, so a
`mlsweep run` typed by the agent brings its trace along:

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

### Continued sessions

When Claude Code continues a session in a new one, it can start the new
transcript that begins with a copy of the old one's history. Capture keeps
each event in one trace. The new trace leaves out the events the old trace already
has, and its header says `continues` with a link to the old one, which says
`continued in` back. The new trace's event numbers start where its own events start.
This applies to traces captured from now on. Traces uploaded before keep any copies
they have.

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

Links are only ever made this way, by the submission itself. There is no
command or button that links a trace to an experiment by hand.

## Reading traces

The Traces tab lists the sessions every linked machine uploaded, newest
activity first. The sidebar filters by agent and by state (live, closed,
imported), and its search box searches every trace's text, title and working
directory; a second box filters the open trace's events. A trace too far down
the list is reached with "Show more traces". The Ctrl+K search palette finds traces
too, by the same text; `trace:WORDS` searches only traces.

A trace page is a ledger: a time column, the event tag, then the content —
your prompts, the agent's replies, its thinking collapsed and dimmed, tool
calls one line each with their result's status and duration. Clicking an event
opens the inspector with the full text and its metadata (model, tokens), and a
copy button. The bar above the ledger shows one kind of event at a time
(user, assistant, thinking, tools, or mlsweep, each with its count). The
mlsweep kind is the tool calls that run the mlsweep CLI, shown as just those
invocations. Its steppers move between prompts (`[` and `]`) and between
mlsweep commands (`m` and `M`). `j`/`k` or the arrow keys move through events,
`Enter` opens one, `Esc` closes the inspector, and `/` jumps to the search box.
The page's URL names the open trace, the selected event and the kind shown, so a
copied link comes back to them.

A trace records the experiments it submitted, and each chip links to one.
Past three, the rest fold behind `+N more`. A chip's `×` removes a link that
is wrong. The `⋮` menu copies the trace id or its
`mlsweep:trace/<id>` reference for a note, downloads the transcript, or deletes
the trace. When notes are served, the header also says how many notes cite the
trace.

An experiment page links to the traces that submitted it, the other way
around too (its Traces entry filters this list to that experiment).

## Reading traces from the terminal

The same traces can be read without the dashboard, which is how an agent
reads them. Every command takes a trace id, a unique prefix of one, or a note
reference, so a link copied from a note works as it is.

```sh
mlsweep capture ls --experiment rzo_e1        # the sessions behind an experiment (--search WORDS too)
mlsweep capture show 9f6c0e22                 # one line per event
mlsweep capture show 9f6c0e22 --kind mlsweep  # only the mlsweep commands it ran
mlsweep capture show "mlsweep:trace/9f6c0e22#E412"   # one event in full
mlsweep capture search warmup "cosine decay"  # matching events across traces
mlsweep capture export 9f6c0e22 -o t.json     # the trace as ATIF
```

`show` prints the trace's header, then one line per event with its number,
time, tag and text, the way the ledger lists them. A tool call's line ends with
its result (`✓ 1.5 s` or `✗`), and an mlsweep command shows only its mlsweep
part. Output stops at about 2000 tokens. Change that with `--budget TOKENS`, and
the last line gives the command for the next page (`--from N`). `--kind` takes
the same kinds as the bar above the ledger. `--event N`, or `#EN` on the
reference, prints one event in full with its metadata and its reference.

`search` matches every word, or a `"quoted phrase"`, against each event's
text, best-matching traces first. Each hit comes with its
`mlsweep:trace/<id>#E<n>` reference, ready to cite. `--trace REF` searches one
trace, and `--kind` narrows the hits as it does for `show`.

`export` writes ATIF, the Agent Trajectory Interchange Format other agent tools
read. Prompts become user steps, and each response of the agent becomes an
agent step with its thinking, reply, tool calls and their results. Since
capture keeps no tool output, a result holds only its status and duration.
Each step lists the event numbers it came from. `--format jsonl` gives the
events as they were uploaded, like the Traces tab's Download transcript. The
`⋮` menu and a right-click on a trace also offer Download as ATIF.

Every command takes `--json` for the data instead of the text.

For a question none of the commands answers, `mlsweep sql` runs one read-only
query over the manager's database, and with no query it lists the tables. The
`capture_events` table holds each event as JSON in its `data` column.

## Notes

A note can link to a trace with the evidence-link syntax:

```markdown
The sweep came out of [that idea about warmup](mlsweep:trace/9f6c0e22).
```

A link can also open one event, or show one kind of event:

```markdown
It submitted [the second sweep](mlsweep:trace/9f6c0e22#E412).
Every [mlsweep command it ran](mlsweep:trace/9f6c0e22?kind=mlsweep).
```

`#E412` is event 412, and the trace opens there with the inspector showing it.
`?kind=` takes user, assistant, thinking, tools or mlsweep. The inspector's
**Copy note link** button gives a link to the selected event, and the `⋮`
menu's reference keeps the kind the ledger is showing.

`mlsweep notes lint` checks that the trace exists on the manager, like every
other `mlsweep:` link, and that the event a link names is in it. (Yes, a trace shows the very session writing the link
that points at itself.)

## Importing history instead

To bring in the transcripts you already have:

```sh
mlsweep capture import --days 30     # --all for everything, --dry-run to look
```

Each imported file becomes a trace marked `imported`, and is never imported
twice. A session the watcher also follows keeps its live state.
