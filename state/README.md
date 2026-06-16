# state/ — the bot's evolving memory (committed data, not code)

Everything here is **data the bot reads and writes**, version-controlled so every change
is auditable as a git diff. The engine code lives in `engine/`; this directory holds the
*evolving* strategy population and run state.

Planned contents (created as later milestones land):

- `population.db` — SQLite: candidates / challengers / champions and their validation results
- `champions.json` — the live "brain": current champions + allocation
- `trial_ledger.json` — cumulative number of trials ever attempted (deflates DSR/PBO honestly)
- `paper_ledger.parquet` — paper/live fills and P&L
- `reports/` — equity curves, DSR/PBO, rolling Sharpe

## KILL_SWITCH

Create a file named `KILL_SWITCH` in this directory and commit it to **immediately halt
all live order placement** — every executor run flattens and stops before trading. Delete
the file (and commit) to resume. This requires no infrastructure: it works from any device
that can push a commit.
