# Notes

A shared folder for research notes, papers, plots and anything else, kept in a
git repository the manager hosts. Agents on any machine clone it and use git as
usual: pull, commit, merge, push. The dashboard's Notes tab browses and edits
the same repository.

Nothing about the files is required: no frontmatter, no folder layout, no
special syntax. If you use them, `[[wikilinks]]`, embeds, math, callouts, charts,
tables and evidence links render, and experiment IDs written anywhere link to
their results. See [Markdown](#markdown).

mlsweep does not run agents and does not decide what goes in the notes. It
provides the remote, the lint, and the tab.

Where the notes live. The manager serves `<mlsweep-dir>/notes`
(`~/.mlsweep/notes` by default) and creates it, as a git repo with the starter
files, the first time it starts. `--notes DIR` serves another repo; `--no-notes`
turns the tab and the remote off. Every `mlsweep notes` command works on the folder you
name, else `$MLSWEEP_NOTES`, else the clone you are in, else that folder. Falling back to
that folder prints a notice, and commands that change files (`mv`, `restore`, `lint --fix`)
refuse to fall back. Name the folder to change it directly.

## How it fits together

```
           manager: ~/.mlsweep/notes
           Notes tab, and git remote at /notes.git
          /            |              \
  agent box A     agent box B     laptop
  clone           clone           clone
```

- The manager's folder is the remote. A push updates its files in place, so the
  tab shows it straight away. Saves from the tab become commits a few seconds later,
  and a push commits any pending saves first.
- Agents do their own merges. If a push is refused because someone else got
  there first, the agent pulls, resolves any conflicts, and pushes again.
- The remote keeps history safe. It refuses force pushes and branch deletes, so
  everything stays recoverable, and checks incoming objects. A pushed symlink arrives
  as a plain file, so it cannot point outside the folder.
- It uses the manager's token. Anyone who can use the manager can push.

`sources/` is a convention, written in `AGENTS.md`: add originals there, and don't
change them. The tab marks those files and lint skips them, but nothing refuses a
change, and the history keeps every version.

## Setup

One machine: start the manager and open the Notes tab. `~/.mlsweep/notes` is
created for you. For an agent on that machine, make a clone as below, because the
manager's folder has the tab and pushes writing into it.

Any machine that can reach the manager:

```sh
mlsweep notes clone ~/notes --manager http://MANAGER:7891
cd ~/notes
git pull      # before you start
git add -A && git commit -m "What I found" && git push
```

The token comes from `--token`, `$MLSWEEP_TOKEN` or `~/.mlsweep/manager.token`. The
clone keeps a copy in `.git/mlsweep-token`, readable only by you, and a credential
helper answers git with it. So plain `git pull` and `git push` need nothing more.

Cloning again into the same folder is safe: it leaves your files alone, refreshes the
token, and applies any new `--name` or `--email`. Agents sharing a machine should each
have their own clone and `--name`.

Plain `git clone http://MANAGER:7891/notes.git` works too: git asks for a password,
which is the token, and the username doesn't matter. Any clone also opens in Obsidian
as a normal vault.

The manager speaks plain HTTP, so use it on a network you trust, or put TLS in front
of it, as for the rest of the manager.

## The Notes tab

### Layout

The left sidebar has Files, Search, Changes and Lint. The centre holds one or more
groups, each with its own tabs, toolbar and open file. The right panel shows
details about the file in the group you last clicked into. Drag the edges to
resize the sidebars. The toolbar buttons at the far left and far right hide them (`L` and `R`).
Groups, their tabs and views, folder state and sizes are remembered per browser.
`notes.html?path=FILE` links to a file, and `?q=WORDS` or `?mentions=EXPERIMENT`
opens a search.

The git status is at the top right and bottom left: click either to commit pending
saves now, and hover the bottom one for the clone URL and current commit.

### Files

- Open: double-click a file to open it in a new tab. A single click only selects
  it, so the tab you are reading stays put. Ctrl+click and middle-click also open
  a new tab.
- Create: right-click a folder, or empty space in the file tree, for New note or New folder. A note name
  may include folders (`ideas/Depth plan`). Clicking a link to a note that doesn't
  exist creates it in the same folder as the note you're in.
- Rename: `F2`, double-click the title in the toolbar, or right-click. Moving
  or renaming rewrites links to the file, in any note.
- Move: drag onto a folder, or right-click → Move to.
- Delete: right-click or the toolbar's `⋯` menu. It stays in git history.
- Upload: the upload icon, or drop files. Dropped on a folder in the tree, they
  go there; dropped elsewhere, beside the open note, and if you're editing, an embed
  is inserted at the cursor. Pasting an image while editing saves it to
  `attachments/` beside the note and embeds it. Uploads are limited to `max_file_mb`.
- Right-click also has Download, History, Copy path and Copy wikilink. Dragging a
  file from the tree into Source inserts a link to it.
- A lock marks folders of originals. A dot marks a file another machine changed in
  the last 15 minutes.

What opens how: markdown in Live, with Live, Source and Split in the toolbar; text
and code in an editor with line numbers (no mode switch); images and PDFs in place;
anything else, or a file over 8 MB, as a download card.

### Tabs and the toolbar

Close a tab with `X`, its ×, or a middle-click. Right-click a tab to close others or
all in its group, split it off, reveal it in the tree, or copy its path. `{` and `}`
switch tabs within the group; `[` and `]` (or the arrows) go back and forward through
what you've opened.

### Splitting and arranging

Drag a tab to arrange the centre:

- Along a tab strip, to reorder it, or onto another group's strip to move it there.
- Onto the edge of a group, to split that group and put the tab in the new half. A
  highlight shows where it will land.
- Onto the middle of another group, to move it there.

`\` splits right and `|` splits down, as do Split right and Split down in the tab's
menu and in `⋯`. A file is open in one group at a time, so splitting moves the tab.
Splitting the only tab in a group leaves an empty group beside it to open something
in. Alt+click a file in the tree, or choose Open to the side, to open it in the group
to the right. Dragging a file from the tree works like dragging a tab, except over a
text editor, where it inserts a link.

A group closes when its last tab closes or moves away. Close an empty group with its
× or `X`. Drag the line between two groups to resize them, and double-click it to
make them equal. Clicking into a group makes its file the one the right panel,
the keys and the `⋯` menu act on.

The `⋯` menu has Rename, Move, Reveal in file tree, Show history, Local graph, Find
mentions of this note, Copy path, Copy wikilink, Download, line width, and Delete.

### Live view

The default for notes. The note is rendered; click a block to edit its markdown in
place, and it renders again when you press Esc or click elsewhere. Blocks you don't
touch stay byte-for-byte as they were.

| In a block | |
|---|---|
| Esc, or click elsewhere | done |
| ↑ on the first line, ↓ on the last | into the block above or below |
| Ctrl+Enter | new block below |
| Backspace in an empty block | remove it |
| `[[` | link completion (↑/↓, Enter or Tab, Esc) |
| Ctrl+B, Ctrl+I | bold, italic |
| Tab | two spaces |

Outside a block: links and experiment chips act on a click (Ctrl+click for a new tab,
Shift+click a chip for its Logs), checkboxes tick, and clicking anything else edits
the block, images included. Use "+ click to add a block" at the end of a note, or
Ctrl+Z / Ctrl+Shift+Z to undo.

A block is: a run of lines up to a blank line; a whole list; a fenced code block; a
`$$` math block; a `:::` or `<div>` container; or the frontmatter. Each renders on its
own, which is the one difference from Split (see [Markdown](#markdown)). A chart or
table block is a block too: click its header to edit the spec, click the chart itself
to zoom or hover, and use the header's `↻` to refetch.

### Source and Split views

Source is the whole file in the same editor used for text and code files: monospace,
line numbers, markdown colored by syntax. Frontmatter, links and embeds, callouts,
footnotes, evidence links, tags and tables are colored too. Enter continues lists, brackets close
themselves, `[[` or Ctrl+Space completes links, Ctrl+B and Ctrl+I format. `E` switches
Live and Source; Esc returns to Live (in a text file, it just leaves the editor). `S`
shows source beside a rendered preview that follows as you type and scroll; click an
image there to enlarge it. These views are for notes; other text and code files always
use this editor.

### Terminals

By default, the tab can open shells on the manager's machine; `mlsweep manager
--no-terminals` turns them off. Open one with the terminal icon after a group's
tabs, or Open terminal here when you right-click
in the file tree, which starts it in
that folder of the notes. Right-click the icon for your commands and for terminals
already running.
A terminal is a tab like any other, so it splits and moves the same way. Nerd Font
icons, as file managers and prompts such as yazi and starship draw, come with the
dashboard, so they show whichever machine the browser is on.

The manager owns each shell. Closing the browser leaves it running, and the tab comes
back attached to it, with what it printed meanwhile. Closing the tab ends the shell,
and so does `exit`, after which any key closes the tab. A manager restart ends every
shell.

Each shell runs in a bubblewrap sandbox, with no container. Its home is the
workspace, `<mlsweep-dir>/workspace`, which persists, shown at your real home path,
so a script that uses `~/...` looks in the same place it would outside. `~/notes` in
it is a clone of the notes, set up and given the token for you, so `git pull` and
`git push` work there, and a new terminal starts in it. That clone can be behind
the manager's copy; if the folder you asked for is not in it yet, the terminal starts
in the nearest one that is and says to `git pull`. `mlsweep` is the manager's own install, read-only, and
`MLSWEEP_MANAGER` and `MLSWEEP_TOKEN` already point it at the manager, so sweeps
submitted from here run on the workers as usual.

`/usr` and `/etc` are read-only, and nothing else from the machine is there unless
you list it in `terminals.json` (below). That leaves out the rest of your home (the
token file, `~/.ssh`), the experiments and the database. There are no GPUs either.
The GPU device files are not in the sandbox, so nothing in it can open one, whatever
it sets. The network is shared, which lets the shell reach the manager on loopback
and an agent reach what it needs.

When `systemd-run --user` works, each shell gets its own scope with a memory cap, a
process limit and a low CPU weight, so it can't starve the manager. Otherwise it runs
under `nice`.

The shell holds the manager's token. Anyone who can open the tab already holds it, so
an agent in a terminal can do what its user can, deleting experiments included.

Terminals need `bwrap` (bubblewrap) on the manager's machine, and a system that
allows unprivileged user namespaces. If either is missing, the icon is greyed out and
says why.

#### terminals.json

`<mlsweep-dir>/terminals.json` says what else the sandbox has. Every key is
optional, and the file is read again for each new terminal, so a change applies to
the next one you open:

```json
{
  "tools": ["~/git/Scripts", "~/.local/bin", "~/.opencode/bin"],
  "mounts": ["~/data/papers"],
  "env": ["OPENAI_API_KEY", "EDITOR=vim"],
  "commands": { "claude": "claude", "pi": "pi" },
  "shell": "/bin/bash",
  "limits": { "memory": "50%", "processes": 1024, "terminals": 8 }
}
```

| Key | |
|---|---|
| `tools` | folders of programs, visible read-only at the same path and on `$PATH` in this order |
| `mounts` | files or folders visible read-only at the same path, not on `$PATH` |
| `env` | `NAME` passes a variable through from the manager's environment; `NAME=value` sets one |
| `commands` | what a new terminal can start with, from the icon's menu; the shell stays open after |
| `shell` | the shell to run (default `/bin/bash`) |
| `limits` | per terminal: `memory` (a size such as `8G` or `50%`, or `null`) and `processes`; and how many `terminals` can be open |

`~` in a path means your home. A symlink in a `tools` folder brings what it points
into: the folder its target is in, the npm package it belongs to, or the whole
prefix around a `bin/` folder, as for a Python that `uv` installed. So
`~/.local/bin` alone is enough for most launchers there. A path that does not exist,
or a variable that is not set, is named in yellow when the terminal opens, and the
rest still works. A mistake in the file itself, such as a misspelled key, stops new
terminals with a message saying what to fix.

Logins are your own business inside the sandbox. Run an agent's login there once,
and it stays in the workspace.

### Saving and conflicts

Saving is automatic about a second after you stop typing (Ctrl+S to do it now), and
every save is checked against the version you opened. A few seconds later it becomes
a commit, named after the manager repo's git identity.

If a push changes the open file: with no unsaved edits it simply updates, and an
open block in Live stays open when the change is elsewhere in the note. With unsaved
edits, nothing is overwritten and a banner offers Compare (a line diff, with Keep
both), Keep mine and Take theirs. If the file was deleted elsewhere, you can
save your copy.

### Right panel

For a note: the experiments it cites with their status, its outline, backlinks with
the linking line, outgoing links (missing ones create the note on click), and history.
For other files: size, modified time, download, the notes linking to it, and history.
With no file open: folder totals and the most cited experiments.

### History and Changes

**Changes** lists every commit from every machine, by author and day. Click a commit
for its full diff, a file to open it, or "diff" beside it for that file's change.
A note's History follows renames. A commit's diff has **View this version** and
**Restore this version**, which writes the old text back as a new change.

An author name is whatever that clone's git config says; nothing verifies it.

### Search

`/` searches file paths and the text of markdown and text files. PDFs are left out. Every word
must match, case-insensitively; `"quote"` a phrase. Click a result to open the file,
or a line to open it at that line. A note opens in Live view, scrolled to that line, with
the search words highlighted for a few seconds.

### Graph

`G` shows notes and their links, colored by top-level folder, sized by link count.
Experiments are orange squares. `Shift+G` (or Local graph in `⋯`) shows the open note's
neighborhood, with a depth slider.

Drag to pan, scroll to zoom, double-click to reset, drag a node to move it, and hover to
highlight its links. Click to open, Ctrl+click for a new tab, or click an experiment
for its Results. The panel filters by path, toggles experiments, images and other
files, and notes not written yet, and sets how many labels show.

### From the rest of the dashboard

The Experiments page counts the documents citing an experiment in its `View ▾` menu,
next to Notes, and those citing a run in the run's `View ▾` menu. Every entry there
shows what it leads to (results and export: finished runs, logs: every run, artifacts:
runs with files). Nothing cites the experiment yet, Notes is there with 0 but greyed
out.

### Keys

Apart from Ctrl+S, everything is a single key that works when you're not typing, so
nothing competes with the browser or your window manager. Esc leaves a text field. `?`
shows this list in the tab.

| Key | |
|---|---|
| Ctrl+S | save now (works while typing) |
| `T` or `O` | open a file |
| `X` | close tab |
| `{` `}` | previous, next tab in the group |
| `\` `\|` | split right, split down |
| `[` `]` | back, forward |
| `E` | notes: Live or Source |
| `S` | notes: Split |
| `/` | search |
| `L`, `R` | hide or show the left, right panel |
| `G`, `Shift+G` | graph, this note's graph |
| `F2` | rename |
| Ctrl+Z, Ctrl+Shift+Z | undo, redo in Live (outside a block) |
| `?` | key list |

In quick open: ↑/↓, Enter to open, Ctrl+Enter for a new tab, Shift+Enter to create
a note with the typed name, Esc to close.

The browser keeps Ctrl+W for itself, so that key closes the browser tab and can't close a note tab.

### Safety

Rendered markdown is sanitized. HTML files are served as text, SVGs sandboxed, and
anything else not an image or PDF as a download, so a file in the notes can't run script
in the dashboard. Paths can't leave the folder or reach `.git`, and symlinks leading
out are not followed. Charts and tables are drawn from the manager's read-only
metrics; a note can choose what to plot but cannot make the page run code or
change a run.

## Markdown

The tab renders GitHub-flavored Markdown (marked 15) with the additions below. Agents
writing with their file tools use the same syntax. Files stay plain markdown, and the
wikilinks, embeds, callouts, footnotes and captions also work in Obsidian. The chart
and table blocks and `mlsweep:` evidence links are dashboard additions: elsewhere
they read as a code block and an ordinary link.

### Base

Headings, emphasis, lists, blockquotes, links, images, inline code, fenced code, rules,
`~~strikethrough~~`, tables with alignment, task lists (tickable), and bare URLs as
links. A single newline doesn't break a line; end it with two spaces, or leave a blank
line for a new paragraph.

### Links

| Write | Result |
|---|---|
| `[[Note]]` | link by file name; `.md` optional, case-insensitive |
| `[[Note\|shown text]]` | with other text |
| `[[Note#Heading]]`, `[[#Heading]]` | to a heading, here or in another note |
| `[[dir/Note]]` | by path from the top; a wrong path falls back to the name |
| `[[paper.pdf]]` | any file, by full name |
| `[text](../dir/note.md)` | relative link; encode spaces as `%20` |
| `[text](/dir/note.md)` | from the top of the folder |
| `[text](#heading-name)` | to a heading in this note |
| `[text](https://…)` | opens in a new browser tab |

If several files share a name, the one in the note's folder wins, then the shallowest;
lint reports it. A link to a missing note is dashed, and clicking it creates the note.

### Embeds

| Write | Result |
|---|---|
| `![[plot.png]]` | image, PNG, JPEG, GIF, WebP, AVIF, BMP, ICO or SVG |
| `![[plot.png\|300]]`, `![[plot.png\|300x200]]` | with a width, or width and height in pixels |
| `![alt](plots/a.png)` | ordinary markdown image; remote URLs load too |
| `![[paper.pdf]]` | PDF viewer in the page |
| `![[Note]]`, `![[Note#Heading]]` | another note or one section, two levels deep at most |
| `![[data.csv]]` | any other file, as a link |

### Charts

A `mlsweep-chart` fence is a query, drawn in the page from the manager's metrics.
It reflects the latest logged data, so don't paste screenshots into a note.

````
```mlsweep-chart
experiments: baseline, warmup
metric: val_loss
split_by: lr
band: minmax
caption: Validation loss by learning rate. {#fig-loss}
```
````

The body is one `key: value` per line; JSON is accepted too. Only `metric` and one
of `experiments`/`runs` are required. Everything else is optional:

| Key | |
|---|---|
| `experiments` | one or more experiment IDs; all their runs are included |
| `runs` | specific runs, `experiment/run` (or a run ID, with `experiments` naming which experiment) |
| `metric` | the logged metric to plot |
| `x` | the x-axis value (default `step`; any numeric logged key works) |
| `split_by` | color by a hyperparameter, or `_run`, `_status`, `_experiment`; runs with the same value are grouped |
| `band` | with `split_by`, the spread across runs: `minmax` (default) or `std`; `none` hides it |
| `agg` | `mean` (default) or `median` within a group |
| `smooth` | EMA weight, 0–0.99; 0 (default) is raw |
| `log_x`, `log_y` | logarithmic axes |
| `x_min`, `x_max` | plot only this x range |
| `where` | keep runs whose hyperparameters match, e.g. `lr=0.001, optimizer=muon` |
| `points` | most points per line (default 400); the manager downsamples |
| `limit` | most runs to read (default 60) |
| `caption`, `title` | caption below the figure, and an explicit header; without `title` the header gives the metric and grouping, plus the experiments unless the legend already names them |

The header gives the metric and grouping; the legend above the plot names the
individual series, so names are not repeated in both. Names are the experiment's
display name and the run's short name (its ID without the experiment prefix); hover
a legend entry for the full IDs. The header's `↻` refetches and `⤓` downloads the
chart as a PNG. Hover a legend name to dim the others, and click it to hide or show
that line. Drag on the plot to zoom and double-click to reset. Charts read the
manager, so they need it reachable; a block whose experiment or metric is missing
shows the reason in place. Naming experiment IDs in the block also counts as citing
them.

### Tables

A `mlsweep-table` fence is a leaderboard, one row per run, fetched the same way.

````
```mlsweep-table
experiments: baseline
metrics: [val_loss, acc]
columns: [lr, batch_size]
sort: val_loss
order: asc
```
````

It takes `experiments`/`runs`, `where`, `limit` and `caption` from the chart
keys, plus `metrics` (the metric columns; default: the experiment's metric, else the
numeric keys the runs logged), `columns`
(hyperparameter columns; default all), `agg` (`final` by default, or `min`,
`max`, `mean`), `sort` (`metrics` sort on the summary `agg`), and `order`.
`limit` defaults to 300 runs here. List `metrics`, or leave it out to infer them
from the runs. Click a column header to sort by it; the arrow shows the current
order. Run labels are short too (the full ID is in the link tooltip), and they link
to Results (Shift+click for Logs). If you list `columns` and the run names are just
one of those values, the run column is dropped and that parameter cell links to the
run instead, so the same name is not shown twice. Each metric cell's tooltip has
final/min/max/mean, and `⧉` copies the table as markdown in the order shown.

### Evidence links

Tie a number in prose to the metric it came from:

```md
Final validation loss was [0.412](mlsweep:baseline/seed0/val_loss@20000), 8% lower.
```

The note says `baseline/seed0/val_loss@20000`; the tab looks the value up and shows
✓ when it matches (within 0.5%) and ⚠ when it does not, with the logged value in
the tooltip. Click opens the experiment's Results. `@` takes a step (the nearest logged
step at or below it is used), or `final`, `min`, `max`, `mean`, or a percentile
such as `@50%`. A metric name may contain `/` (`train/loss`); the first two
segments are the experiment and run. Use it for every number about a run: it is one click of checking
for the reader. Lint reports references to experiments or runs that do not exist,
and numbers that do not match the logged value. Elsewhere the link is just a link.

### Footnotes

```md
One seed is one seed.[^seed]

[^seed]: Repeat with at least three and report the spread.
  Continuation lines are indented.
```

Footnotes render at the end of the block that defines them, with a backlink.
Numbers are counted across the whole note in Live, so a reference and its
definition can be in different blocks.

### Figures and cross-references

A `Figure:`, `Table:` or `Chart:` line on the line directly below an image or a
table becomes a caption; a chart or table block's `caption:` does the same.
Figures and tables are numbered separately, in document order. End a caption with
`{#fig-name}` to fix its anchor, then link to it from prose as
`[Figure 2](#fig-name)`.

A heading can fix its own anchor too: `## Results {#results}`, linked as
`[the results](#results)`.

### Experiment IDs

Any word that's an experiment ID on this manager becomes a chip with a status dot,
linking to Results, in text or `inline code`. The chip shows the experiment's
display name; hover it for the full ID. Inside a fenced block it stays plain but
still counts as a citation for backlinks, the citation counts and lint.

### Run IDs

A word that's a run ID on this manager counts as a citation the same way, so a note can
talk about one run of an experiment. The Experiments page shows the documents citing an
experiment (its own or any of its runs) beside its name, and the documents citing that
run's `View ▾` menu. A run ID belongs to its experiment, not to the manager: two
experiments can each have a `seed0`, and such a shared ID counts only for the
experiments the same note also names.

### Math

KaTeX. `$x^2$` inline; `$$…$$` or a ```` ```math ```` block for display (a `$$` block may
contain blank lines). A `$` before a space, or a closing `$` before a digit, is not math,
so "$5 and $10" stays text; `\$` is a literal dollar. Invalid TeX shows in red instead of
breaking the note. `\href`, `\url` and `\includegraphics` are disabled.

### Callouts

```
> [!warning] Optional title
> Body, which can hold any markdown.
```

Without a title, the type is the title. Types with their own icon or color: `note`, `info`, `tip`,
`hint`, `important`, `warning`, `caution`, `attention`, `danger`, `error`, `bug`, `failure`,
`fail`, `success`, `check`, `done`, `question`, `help`, `faq`, `example`, `quote`, `cite`,
`abstract`, `summary`, `tldr`, `todo`, `idea`; any other word gets the default look.
Obsidian's `[!note]-` fold marker is accepted, but callouts don't fold here.

### Frontmatter

A `---` block on the first line shows as a properties box, one row per `key: value`
line. Nothing reads the values; add them or not.

### Code

Fences highlight `python`, `sh`/`bash`/`zsh`/`shell`/`console`, `js`, `ts`, `json`, `yaml`,
`toml`, `diff`/`patch`, `c`, `cpp`, `cu`/`cuda`, `java`, `rs`/`rust`, `go` and `md`. Other
languages show as plain text with their name in the corner.

### HTML

Inline HTML works (`<details>`, `<kbd>`, `<sub>`, …) after sanitizing. Removed: `<script>`,
`<style>`, `style=` and event attributes, `<form>`, `<button>`, `<textarea>`, `<select>`,
`<iframe>`, `<object>`, `<embed>`, `<link>`, `<meta>`, `<base>`, inputs other than task
checkboxes, comments, and `javascript:` links. An HTML comment is a way to leave a note
that doesn't render.

### Not supported

| Syntax | What happens |
|---|---|
| Pandoc divs, `::: {.cls}` | shown as literal text |
| Other attributes, `# Title {.c}` | literal text; `{#id}` on a heading or caption is the one exception |
| Definition lists, citations `[@key]` | literal text |
| `x^2^` | literal text; use `<sup>` or math |
| `H~2~O` | single tildes are strikethrough; use `<sub>` or math |
| Smart quotes, `--` to dash | left as typed |

In Live, reference-style links (`[x][ref]` with `[ref]: url` in another block) don't
resolve, because each block renders alone; they do in Split. Use inline links if you
work mainly in Live.

## Lint

```sh
mlsweep notes lint ~/notes            # report
mlsweep notes lint ~/notes --fix      # also apply the safe fixes
mlsweep notes lint ~/notes --json
```

Three tiers:

1. Safe fixes: a link whose path is wrong but whose file name matches exactly
   one file is pointed at that file. Nothing else is ever changed for you.
2. Reports (suspects, for a person or an agent to judge):
   - broken and ambiguous links;
   - experiment IDs the manager does not know, including those in chart/table
     blocks and `mlsweep:` evidence links, and run IDs it does not know;
   - malformed chart/table specs and unknown keys;
   - evidence links whose number does not match the logged metric;
   - leftover conflict markers and empty files.
3. Judgment is not automated. Lint checks that an evidence link's experiment and
   run exist and that the number beside it matches the logged value; it does not
   decide whether that is the right number to cite, or whether a claim follows
   from it. Whoever writes a note still reads the curve. `mlsweep notes lint
   --review-prompt` (or *Copy review prompt* in the tab) prints a brief for a
   fresh agent: check the notes' claims against the logged metrics and sources,
   look for contradictions and over-broad conclusions, and write findings to a
   new file under `reviews/` without editing anything. Give it to an agent that
   did not write the notes.

Files under `sources/` are originals and are not linted.

The experiment/run ID and evidence-number checks need the manager (`--manager`,
`MLSWEEP_MANAGER`, default `http://localhost:7891`, and its token as usual).
Without it, lint still parses the chart/table specs, checks links and reports
merge leftovers; it leaves the numbers unchecked.

## Rules for agents

The starter `AGENTS.md` in the folder says this; repeat it in your agent's
instructions if you like. It assumes notes work when a request is open-ended, and
does whatever else it is asked.

Always:

- Pull before starting, then commit and push as you go, always before you stop. If a push
  is refused, pull, resolve any conflicts so both sides' work survives, and push again. Never force-push.
- Read before writing: `grep -ri` the folder for what is known.
- Prefer a new file to rewriting someone else's. To correct a note, add a dated
  line or a new note that links to it.
- `sources/` is read-only except for adding files.
- Keep `STATUS.md` short and readable by someone with no context: what is running, with
  experiment IDs, and what comes next. Read it when you start, and update it before you stop.

When a note reports experiment results:

- Every number comes from the experiment's logged metrics, and the note names the
  experiment and run. Write it as an evidence link so a reader can check it; lint
  reports a number that disagrees. Plot with a chart block instead of a screenshot,
  and state the conditions (steps, seeds, size).
- Record dead ends, with the reason.

## Configuration

`.mlsweep-notes.json` in the repo:

```json
{ "protected": ["sources/"], "max_file_mb": 50 }
```

`protected` lists originals, which the tab marks and lint skips. A pattern ending in
`/` names a folder; anything else is a glob on the path. `max_file_mb` limits uploads
from the tab.

## The tab and the command line do the same things

Everything the Notes tab does has a command, so an agent can do it too, and the
other way round:

| In the Notes tab | Command |
|---|---|
| read, edit, create, upload, delete | the agent's own file tools and git on its clone |
| rename / move (links follow) | `mlsweep notes mv SRC DST` |
| search | `mlsweep notes search WORDS` (or `grep -ri`) |
| side panel: backlinks, outgoing links, cited experiments | `mlsweep notes links PATH` |
| Experiments page citation counts | `mlsweep notes mentions EXPERIMENT` (its runs count too) |
| Changes feed, file history | `mlsweep notes history [PATH]` |
| a commit's diff, view an old version | `mlsweep notes diff REV [PATH]`, `mlsweep notes show REV PATH` |
| restore an old version | `mlsweep notes restore PATH REV` |
| Lint, safe fixes, review prompt | `mlsweep notes lint [--fix] [--review-prompt]` |
| Commit button, git status | `git commit`, `mlsweep notes status` |

Only the graph has no command; `links` gives the same neighbors as text.

## Command reference

| Command | |
|---|---|
| `mlsweep notes clone DIR [--manager URL] [--name N] [--email E]` | clone from the manager |
| `mlsweep notes status [DIR]` | remote, unpushed commits, recent changes |
| `mlsweep notes lint [DIR] [--fix] [--json] [--review-prompt]` | checks |
| `mlsweep notes mv / links / mentions / search / history / show / diff / restore` | see the table above |
| `mlsweep manager [--notes DIR \| --no-notes] [--no-terminals]` | serve the tab and the git remote; `--no-terminals` turns terminals off |
