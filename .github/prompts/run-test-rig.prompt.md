---
name: run-test-rig
description: 'Run Happy-Hare/Klipper on the test rig using the VS Code task "Run Test on Test Rig" and report runtime status/errors.'
argument-hint: 'Optional: include what to validate after launch (for example BLDC startup, sync modes, or specific log checks).'
---
Run the code on the test rig using the VS Code task named `Run Test on Test Rig`.

Procedure:
1. Execute the task `Run Test on Test Rig` in the `klipper` workspace folder.
2. Wait for completion and capture whether it succeeded or failed.
3. If it fails, summarize the key error lines from task output.
4. If it succeeds, report success and any immediate warnings worth attention.
5. If the user included additional validations in the prompt arguments, perform those checks and report results.

Output format:
- Task run result: success/failure
- Key output summary
- Errors/warnings (if any)
- Follow-up recommendation (if needed)
