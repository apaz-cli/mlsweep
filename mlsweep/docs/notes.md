# Notes

A shared folder for research notes, papers, plots and anything else, kept in a
git repository the manager hosts. Agents on any machine clone it and use git as
usual: pull, commit, merge, push. The dashboard's Notes tab browses and edits
the same repository.

Nothing about the files is required: no frontmatter, no folder layout, no
special syntax. If you use them, `[[wikilinks]]`, embeds, math and callouts render, and
experiment IDs written anywhere link to their results. See [Markdown](#markdown).

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

The left sidebar has Files, Search, Changes and Lint. The centre has note tabs and
the open file. The right panel shows details about that file. Drag the edges to
resize the sidebars; the toolbar button at the far right hides the right panel.
Open tabs, their views, folder state and sizes are remembered per browser.
`notes.html?path=FILE` links to a file, and `?q=WORDS` or `?mentions=EXPERIMENT`
opens a search.

The git status is at the top right and bottom left: click either to commit pending
saves now, and hover the bottom one for the clone URL and current commit.

### Files

- Open: click a file. It replaces the current tab unless that tab has unsaved
  changes. Ctrl+click or middle-click opens a new tab.
- Create: `N`, the icons above the tree, or right-click a folder. A note name
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

What opens how: markdown in Live; text and code in an editor with line numbers;
images and PDFs in place; anything else, or a file over 8 MB, as a download card.

### Tabs and the toolbar

Close a tab with `X`, its ×, or a middle-click. Right-click a tab to close others or
all, reveal it in the tree, or copy its path. `{` and `}` switch tabs; `[` and `]`
(or the arrows) go back and forward through what you've opened.

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
own, which is the one difference from Split (see [Markdown](#markdown)).

### Source and Split views

Source is the whole file in one editor with markdown colored by syntax. Enter continues
lists, brackets close themselves, `[[` or Ctrl+Space completes links, Ctrl+B and Ctrl+I
format. `E` switches Live and Source; Esc returns to Live (in a text file, it just
leaves the editor). `S` shows source beside a rendered preview that follows as you
type and scroll; click an image there to enlarge it.

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
or a line to open it at that line.

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
| `N` | new note |
| `X` | close tab |
| `{` `}` | previous, next tab |
| `[` `]` | back, forward |
| `E` | Live or Source |
| `S` | Split |
| `/` | search |
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
out are not followed.

## Markdown

The tab renders GitHub-flavored Markdown (marked 15) with the additions below. Agents
writing with their file tools use the same syntax. Files stay plain markdown, and the
wikilinks, embeds and callouts also work in Obsidian.

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

### Experiment IDs

Any word that's an experiment ID on this manager becomes a chip with a status dot,
linking to Results, in text or `inline code`. Inside a fenced block it stays plain but
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
`toml`, `diff`/`patch`, `c`, `cpp`, `cu`/`cuda`, `java` and `md`. Other languages, including
Rust and Go, show as plain text with their name in the corner.

### HTML

Inline HTML works (`<details>`, `<kbd>`, `<sub>`, …) after sanitizing. Removed: `<script>`,
`<style>`, `style=` and event attributes, `<form>`, `<button>`, `<textarea>`, `<select>`,
`<iframe>`, `<object>`, `<embed>`, `<link>`, `<meta>`, `<base>`, inputs other than task
checkboxes, comments, and `javascript:` links. An HTML comment is a way to leave a note
that doesn't render.

### Not supported

| Syntax | What happens |
|---|---|
| Footnotes, `x[^1]` | not rendered as footnotes; `[^1]: text` is read as a link definition, so `[^1]` becomes a stray link |
| Pandoc divs, `::: {.cls}` | shown as literal text |
| Attributes, `# Title {#id .c}` | shown as literal text |
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
   - experiment IDs the manager does not know;
   - leftover conflict markers and empty files.
3. Judgment is not automated, and that includes numbers. Lint doesn't compare
   them with logged metrics, so whoever writes a note checks each one with
   `mlsweep metrics`. `mlsweep notes lint --review-prompt` (or *Copy
   review prompt* in the tab) prints a brief for a fresh agent: check the notes'
   claims against the logged metrics and sources, look for contradictions and
   over-broad conclusions, and write findings to a new file under `reviews/`
   without editing anything. Give it to an agent that did not write the notes.

Files under `sources/` are originals and are not linted.

The experiment ID check needs the manager (`--manager`, `MLSWEEP_MANAGER`, default
`http://localhost:7891`, and its token as usual); without it, lint still runs the
other checks.

## Rules for agents

The starter `AGENTS.md` in the folder says this; repeat it in your agent's
instructions if you like:

- Pull before starting, then commit and push as you go, always before you stop. If a push
  is refused, pull, resolve any conflicts so both sides' findings survive, and push again. Never force-push.
- Read before writing: `grep -ri` the folder for what is known.
- Prefer a new file to rewriting someone else's. To correct a note, add a dated
  line or a new note that links to it.
- `sources/` is read-only except for adding files.
- Every number about an experiment comes from its logged metrics, and the note
  names the experiment ID. Check each one with `mlsweep metrics`, and state the
  conditions (steps, seeds, size).
- Record dead ends, with the reason.
- Keep `STATUS.md` short and readable by someone with no context: what is running, with
  experiment IDs, and what comes next. Read it when you start, and update it before you stop.

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
| `mlsweep manager [--notes DIR \| --no-notes]` | serve the tab and the git remote |
