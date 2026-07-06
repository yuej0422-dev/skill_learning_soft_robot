# PLAN

1. Record the current workspace and Python/LeRobot/PyTorch environment.
2. Create a new `soft_vla` project folder with configs, source modules, scripts,
   tests, outputs, and reports.
3. Define the fixed soft-robot interface:
   - 13D state with binary `state[12] == gripper_state`.
   - 7D action with TCP delta in dimensions 0..5 and binary
     `action[6] == gripper_action`.
4. Implement a synthetic visual servoing task with three synchronized RGB
   cameras, deterministic dynamics, bounded noise, and non-empty language tasks.
5. Write the synthetic dataset through the installed LeRobot 0.4.4
   `LeRobotDataset.create/add_frame/save_episode` API when available.
6. Implement dataset inspection and reports:
   - feature names, shapes, units, camera keys, episode counts,
   - state/action dimensionality,
   - binary gripper validation,
   - timestamp monotonicity,
   - velocity and delta-action consistency checks.
7. Add SmolVLA adapter scripts that probe the installed LeRobot API and fail
   clearly if `transformers`, `peft`, or GPU Torch are missing.
8. Implement dry-run offline inference using dataset replay,
   `SafetyFilter`, `ActionBuffer`, and `NullRobotController`.
9. Add unit tests for schema, feature mapping, safety filtering, null hardware,
   and the offline pipeline.
10. Run the parts supported by the available environment and record the
    remaining dependency gaps in `reports/environment.md` and
    `FINAL_REPORT.md`.

