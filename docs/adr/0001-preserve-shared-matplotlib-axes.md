# Preserve shared Matplotlib axes across plot rebuilds

GerryTools plot builders treat a bound Matplotlib `Axes` as a shared surface. Rebuilds remove only
artists created by GerryTools, preserve external artists and per-setting Matplotlib changes, and
apply most-recent-wins ownership independently to each managed setting. Explicit GerryTools
settings may reclaim a setting and survive `bind_to_ax()`, while per-axes history resets on rebind.

This contract requires artist tracking, per-setting snapshots and ownership, restoration of
externally owned limits after autoscaling, and a second reconciliation after annotations that can
change limits. The resulting state-management implementation is intentional: `ax.clear()`, one
global axes owner, and eager-only rendering would be simpler but would discard caller-owned
content or break the last-writer behavior.
