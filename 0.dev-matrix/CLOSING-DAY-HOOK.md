# Closing Day Hook

Run `npm run close-day` from repo root before ending the work day, before push, or before claiming readiness.

Start the session with `0.dev-matrix/resume-work.ps1` when it exists so background launch-check begins early instead of waiting for close-day.

The hook keeps close-day short: it reuses the latest background launch-check status from `resume-work.ps1`, records `git status`, and writes `0.dev-matrix/LAST-CLOSEOUT.md`. Heavy verification should be done during active work or an explicit readiness pass, not deferred until close-day.

Close-day or task-complete handoff should also surface a project progress snapshot from `0.dev-matrix/project-progress.ps1` when it exists:
- `Date:` today's date for the snapshot
- `Working since:` first recorded repo work date
- `Working days:` elapsed days since first commit
- `Completion:` exact completion percentage plus completed/total task counts when `TASK.md` supports it
- `Pending days at current pace:` projected days remaining when progress can be computed
- `Next:` the next 3 concrete tasks to move the project forward

If the repo does not yet have a usable `TASK.md`, close-day should report completion as unavailable and explicitly call out that the task board must be added before exact project percentage can be reported.

Required close-day handoff fields in the newest `AI-HANDOFF.md` entry:
- `Changed:`
- `Verified:`
- `Operational proof:`
- `Continue from:`
- `Next step:`
- `Blockers:`

Required launch-focus fields in `0.dev-matrix/LAUNCH_CHECKLIST.md`:
- `Product outcome:`
- `Current launch slice:`
- `Current blocker:`
- `Next earning step:`
