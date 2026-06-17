"""Orchestration: the executor that turns champions into orders, and the state it commits.

The same ``run_step`` code path serves paper and live; only the injected broker and the
live-enable gate differ. State (account, fills, equity, champions) is committed to ``state/``
so every decision is auditable as a git diff and the next stateless cron run can resume.
"""
