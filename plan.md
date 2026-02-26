# Viewer Cleanup Plan

## Why this cleanup now

The recent viewer work fixed real bugs and added useful controls, but it also increased complexity in timing, pause behavior, and UI-driven refresh paths. We should stabilize architecture before adding more features so future changes are predictable, testable, and low-risk.

Current pain points:
- State is spread across multiple flags and classes (`BaseViewer`, backend viewer, scene).
- Update behavior differs between running vs paused, and between sim-driven vs UI-driven changes.
- Overlay logic (contacts/debug/rewards/metrics) is tightly interleaved with backend viewer code.
- Regression risk is rising because many fixes are interaction-dependent.

## Cleanup goals

1. Make state flow explicit and centralized.
2. Make update triggers deterministic and backend-agnostic.
3. Isolate overlays from core tick/render loop.
4. Add targeted tests and lightweight invariants for failure paths.

## Execution plan

### Phase 1: State + Actions Cleanup (start here)
- Add a single viewer status snapshot API in `BaseViewer` for UI consumption.
- Remove duplicated status formatting logic from native and Viser paths.
- Normalize action handling so UI callbacks enqueue actions consistently.
- Keep behavior unchanged while simplifying data flow.

Success criteria:
- Both native and Viser status displays read from the same status snapshot.
- No direct cross-thread state mutation from UI callbacks.
- Existing tests continue to pass.

### Phase 2: Update-Pipeline Cleanup
- Introduce explicit update reasons (`SIM_STEP`, `UI_TOGGLE`, `PAUSE_REFRESH`, `ENV_SWITCH`, etc.).
- Centralize refresh gating so paused/live behavior is handled in one place.
- Remove ad-hoc coupling between `_needs_update` and scene-local flags where possible.

Success criteria:
- One deterministic path decides when live recompute vs cached refresh is used.
- Paused toggles (contacts/debug) are covered by dedicated tests.

### Phase 3: Hardening + Instrumentation
- Add targeted tests for failure/recovery and paused toggle flows.
- Add debug-only invariants (e.g., sim-time only advances on successful step).
- Add lightweight timing breakdown hooks for sim/render/update phases.

Success criteria:
- Clear test coverage for the known fragile interaction paths.
- Easier diagnosis when performance/regression issues appear.

### Phase 4: Overlay Modularization
- Extract contacts/debug/reward/metrics update code into small components.
- Keep backend viewers focused on input wiring + scene submission.
- Preserve existing behavior while reducing file-level complexity.

Success criteria:
- Overlay behavior can be modified independently with localized tests.
- Viewer backends become thinner and easier to extend.

## Delivery strategy

- Land this as a sequence of small, reviewable commits.
- Prioritize behavior-preserving refactors before structural extraction.
- Run `make check` and fast tests after each phase-sized change.

## Progress

- Completed:
  - Added a shared `ViewerStatus` snapshot API in `BaseViewer` and switched native + Viser status rendering to consume it.
  - Added deterministic Viser update-policy helpers (`_should_update_cameras`, `_should_submit_scene_update`) plus unit tests.
  - Replaced ad-hoc Viser `_needs_update` boolean with explicit pending update reasons.
  - Refactored `ViserPlayViewer.sync_env_to_viewer()` into focused helper methods (plots, cameras, debug queue, submit).
- In progress:
  - Consolidating scene refresh semantics for paused/live/UI-triggered updates.
- Remaining:
  - Introduce a richer update-reason contract end-to-end (`SIM_STEP`, `UI_TOGGLE`, `PAUSE_REFRESH`, `ENV_SWITCH`).
  - Add explicit invariant checks and richer timing diagnostics in debug mode.
  - Extract overlay modules (contacts/debug/reward/metrics) behind cleaner interfaces.
