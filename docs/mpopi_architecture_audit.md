# MPOPI → PPO: Repository Architecture Audit

Status: Phase 1–3 deliverable. No implementation code has been written.
Audited revision: `c2e1e06` (mjlab 1.6.0), `rsl-rl-lib==5.5.1` (pinned in
`pyproject.toml`; source verified against the `uv.lock` sdist hash
`50129065…fab43`).

RSL-RL line numbers below refer to the 5.5.1 sdist. They will drift if the pin
changes.

## 0. Environment caveat

This audit was produced by reading source only. The machine has no `uv`,
no `.venv` and no CUDA device. Nothing was executed and no tests were run.
Every claim below comes from reading the code, not from runtime observation.

## 1. Current training flow

```
mjlab.scripts.train:main            (tyro CLI → TrainConfig)
  └─ launch_training                (log dir, GPU selection, torchrunx if >1 GPU)
      └─ run_train                  src/mjlab/scripts/train.py
          ├─ ManagerBasedRlEnv(cfg.env)
          ├─ RslRlVecEnvWrapper(env, clip_actions)             train.py:148
          ├─ runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner   :153
          ├─ dump_yaml(params/agent.yaml, asdict(cfg.agent))  :165
          ├─ runner = runner_cls(env, asdict(cfg.agent), log_dir, device)   :167
          │     MjlabOnPolicyRunner.__init__  (strips None cfg keys)  runner.py:16-32
          │       └─ OnPolicyRunner.__init__                  [rsl_rl] on_policy_runner.py:26
          │            ├─ alg_class = resolve_callable(cfg["algorithm"]["class_name"])  :39
          │            ├─ self.alg = alg_class.construct_algorithm(obs, env, cfg, device) :40
          │            │     PPO.construct_algorithm          [rsl_rl] ppo.py:427
          │            │       ├─ actor/critic = resolve_class(cfg["actor"/"critic"])
          │            │       ├─ storage = RolloutStorage("rl", N, T, obs, [A])   ppo.py:455
          │            │       └─ alg_class(actor, critic, storage, **alg_cfg)     ppo.py:458
          │            └─ Logger(...)
          └─ runner.learn(max_iterations, init_at_random_ep_len=True)       :175
                OnPolicyRunner.learn                          [rsl_rl] on_policy_runner.py:56
                  for it:
                    with inference_mode:
                      for T steps:
                        a = alg.act(obs)                      ppo.py:124
                        obs, r, d, extras = env.step(a)       vecenv_wrapper.py:72
                        alg.process_env_step(obs, r, d, extras)  ppo.py:137
                        logger.process_env_step(...)
                      alg.compute_returns(obs)                ppo.py:166
                    loss_dict = alg.update()                  ppo.py:193
                    logger.log(it, ..., loss_dict, ...)
```

## 2. Answers to the eleven audit questions

| # | Question | Location | Owner |
|---|----------|----------|-------|
| 1 | Where PPO is instantiated | `PPO.construct_algorithm`, `rsl_rl/algorithms/ppo.py:427-463`, called from `OnPolicyRunner.__init__` `on_policy_runner.py:39-40`. The class comes from `cfg["algorithm"]["class_name"]` via `resolve_callable`. | RSL-RL |
| 2 | Where the runner is instantiated | `src/mjlab/scripts/train.py:167`. The class is from `load_runner_cls(task_id)` (task registry, `src/mjlab/tasks/registry.py:66`), falling back to `MjlabOnPolicyRunner`. Task-specific subclasses: `VelocityOnPolicyRunner`, `MotionTrackingOnPolicyRunner`, `ManipulationOnPolicyRunner`, all subclassing `MjlabOnPolicyRunner`. | mjlab |
| 3 | Where rollout data is collected | `OnPolicyRunner.learn`, `on_policy_runner.py:82-98` (fixed `num_steps_per_env` loop under `torch.inference_mode`). | RSL-RL |
| 4 | Where obs/actions/rewards/dones are stored | `PPO.act` fills `self.transition` (`ppo.py:124-135`), `PPO.process_env_step` finalises it (`ppo.py:137-164`), `RolloutStorage.add_transition` copies it into preallocated `[T, N, ...]` tensors (`rsl_rl/storage/rollout_storage.py:173-205`). Observations are a `TensorDict` with batch size `[T, N]`. | RSL-RL |
| 5 | Where log-probs are computed | `PPO.act`: `actor.get_output_log_prob(actions)` (`ppo.py:131`), stored as `storage.actions_log_prob [T,N,1]`. Distribution params (`(mean, std)` for Gaussian) stored as `storage.distribution_params` (`ppo.py:132`). Recomputed per minibatch in `update` (`ppo.py:228-234`). The log-prob is summed over action dims (`distribution.py:230-232`). | RSL-RL |
| 6 | Where advantages/returns are computed | `PPO.compute_returns`, `ppo.py:166-191`. Standard GAE(λ) over the `[T, N]` rollout, with bootstrap from `critic(last_obs)`. Advantages are normalised over the whole batch unless `normalize_advantage_per_mini_batch`. | RSL-RL |
| 7 | Where minibatches are constructed | `RolloutStorage.mini_batch_generator` (`rollout_storage.py:225-258`, feed-forward) and `recurrent_mini_batch_generator` (`:261-329`). It flattens `[T,N]` to `[T·N]` and uses one `randperm` shared across epochs. It yields `RolloutStorage.Batch`. | RSL-RL |
| 8 | Where the PPO loss is calculated | Inline in `PPO.update`, `ppo.py:268-285`. Clipped surrogate, clipped value loss and entropy bonus, all reduced with an unweighted `.mean()`. | RSL-RL |
| 9 | Where `optimizer.step()` occurs | `ppo.py:311`, after `clip_grad_norm_` on actor and critic separately (`:309-310`). There is one optimizer over actor and critic parameters (`ppo.py:100-102`). | RSL-RL |
| 10 | mjlab vs RSL-RL ownership | **mjlab:** `src/mjlab/rl/{config,runner,vecenv_wrapper}.py`, `scripts/train.py`, `tasks/registry.py`, task `rl_cfg.py` files, task runner subclasses. **RSL-RL:** runner loop, PPO, `RolloutStorage`, models, distributions, `Logger`. | — |
| 11 | Clean extension points | See §4. | — |

### Additional facts that matter for off-policy work

- **The time-out bootstrap is folded into the stored rewards.**
  `process_env_step` does `rewards += γ · V_{θ_old}(s_t) · time_out`
  (`ppo.py:154-158`). This uses the critic *at collection time* and uses
  `V(s_t)` as a proxy for the unavailable `V(s_{t+1})`. Replaying
  `storage.rewards` would silently reuse stale critic values, so the replay
  path must store **raw rewards and `time_outs` separately** and re-apply the
  bootstrap with the current critic.
- **Auto-reset hides the terminal observation.** With
  `ManagerBasedRlEnvCfg.auto_reset=True` (the default), the observation returned
  on a done step is the post-reset state. A true `s_{t+1}` for terminal or
  truncated steps is **not available** to the learner. That is why RSL-RL
  bootstraps time-outs with `V(s_t)`. A replay "next observation" field would
  therefore be misleading on done steps. The design instead stores contiguous
  `[T, N]` segments plus the segment's final bootstrap observation, which is
  exactly what GAE and V-trace need.
- **`dones` merges terminated and truncated** (`vecenv_wrapper.py:78-81`).
  `extras["time_outs"]` carries the truncation flag only when
  `not cfg.is_finite_horizon` (`:82-83`). `terminated` equals
  `dones ∧ ¬time_outs`.
- **Actions are stored unclipped.** `clip_actions` is applied inside the
  wrapper after sampling (`vecenv_wrapper.py:75-76`), so the stored `log μ(a|s)`
  is the density of the stored action. This is consistent for IS.
- **Gaussian and Beta policies have full support.** Gaussian std is clamped
  to at least `1e-6` (`distribution.py:173-174`). `μ(a|s) = 0` cannot occur
  analytically, but `log μ` can be extremely negative, and Beta at the exact
  boundary gives `-inf`. Guard with finite masks and log-ratio clamps.
- **Observation normalisation.** `update_normalization` runs on *fresh* storage
  only (`ppo.py:328-330`). Raw observations are stored, so replayed observations
  are re-normalised with the current statistics at forward time. This is
  correct: the stored `log μ` was computed with μ's own normaliser, so μ is
  well defined as a function, and π_old is evaluated with π_old's normaliser.
- **The logger already forwards arbitrary `loss_dict` keys** as
  `Loss/{key}` scalars and console lines (`rsl_rl/utils/logger.py`,
  `log()`). Base PPO 5.5.1 does **not** log KL or clip fraction. Those must be
  added.
- **Env steps are already logged.** `Logger.tot_timesteps +=
  num_steps_per_env · num_envs · world_size` per iteration. Replay does not
  change env steps per iteration, so reward vs env steps is recoverable.
- **The adaptive LR schedule** uses `KL(π_old ‖ π_θ)` computed from
  `batch.old_distribution_params` (`ppo.py:241-266`). For replay samples those
  params must be π_old's, recomputed, **not** μ's. Otherwise the schedule would
  react to policy drift across iterations rather than to the current step size.

## 3. Dependency boundaries

```
┌────────────────────────── mjlab (editable) ───────────────────────────┐
│ scripts/train.py ── TrainConfig (tyro) ── asdict(cfg.agent) ─┐        │
│ rl/config.py   RslRl*Cfg dataclasses                         │        │
│ rl/runner.py   MjlabOnPolicyRunner(OnPolicyRunner)  ◄────────┘        │
│ rl/vecenv_wrapper.py  RslRlVecEnvWrapper(VecEnv)                      │
│ tasks/registry.py     task_id → (env_cfg, rl_cfg, runner_cls)         │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ subclass / config dict / class_name string
┌───────────────────────────────▼──── rsl-rl-lib==5.5.1 (do not edit) ──┐
│ OnPolicyRunner.learn   (rollout loop, logging, save)                  │
│ PPO  act / process_env_step / compute_returns / update (loss inline)  │
│ RolloutStorage [T,N] ring, Transition, Batch, mini_batch_generator    │
│ MLPModel / distributions (log_prob, kl_divergence, params)            │
│ utils.resolve_callable("pkg.mod:Class")                               │
└───────────────────────────────────────────────────────────────────────┘
```

## 4. Proposed integration points (smallest clean set)

1. **Algorithm class via `class_name`, no RSL-RL edits.**
   `resolve_callable` accepts `"module.path:Class"` (`rsl_rl/utils/utils.py:129-138`).
   A subclass `mjlab.rl.mpopi:MpopiPpo(PPO)` is constructed by the unmodified
   `PPO.construct_algorithm` (static, uses the resolved class,
   `ppo.py:430,458`). Extra config arrives as a constructor kwarg.
2. **Overrides inside `MpopiPpo`, and nothing else:**
   - `process_env_step`: capture raw rewards and `time_outs` before
     `super()` bakes in the bootstrap. The rest is delegated.
   - `compute_returns`: `super()` (fresh GAE unchanged), then remember the
     bootstrap observation for this segment.
   - `update`: **must be re-implemented.** The loss is inline with an unweighted
     `.mean()` (`ppo.py:268-285`), and there is no hook for per-sample weights or
     for extra samples. The override mirrors the upstream loop and adds
     (a) replay samples, (b) per-sample IS weights, and (c) KL and clip-fraction
     metrics. Upstream drift is controlled by the exact pin plus an
     equivalence test (see Risks).
3. **Mode switch in mjlab's runner.** `MjlabOnPolicyRunner.__init__` already
   rewrites `train_cfg` before `super().__init__` (it strips `None` optional
   configs, `runner.py:24-31`). The same place translates
   `algorithm.mpopi.mode`:
   - `"ppo"`: pop the `mpopi` key so base `PPO` receives *exactly* today's
     kwargs, and leave `class_name` unchanged.
   - `"naive_replay_ppo"` / `"mpopi_ppo"`: set
     `class_name="mjlab.rl.mpopi:MpopiPpo"` and pass `mpopi_cfg=...`.
   All task runners subclass `MjlabOnPolicyRunner`, so every task gets the
   switch.
4. **Config.** A new `MpopiCfg` dataclass nested as
   `RslRlPpoAlgorithmCfg.mpopi` (non-optional, default `mode="ppo"`) gives flat
   tyro flags, e.g. `--agent.algorithm.mpopi.mode mpopi_ppo`.
5. **Replay storage: a new minimal class, not `RolloutStorage` reuse.**
   `RolloutStorage` is a single `[T,N]` rollout that is cleared every iteration.
   Its `Batch` has no weight or version fields. Reusing its *layout*
   (`TensorDict [T,N]`, the same field names) while storing `K` segments in a
   `[K,T,N]` GPU ring avoids per-transition bookkeeping, and keeps trajectories
   contiguous so advantages can be recomputed with the current critic.
6. **Metrics** go through `loss_dict` (logged as `Loss/mpopi/*`). There are no
   logger changes. The prefix is cosmetically wrong, and fixing it would mean
   subclassing `Logger`. That is deferred.

Nothing requires forking or patching RSL-RL.

## 5. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| `MpopiPpo.update` duplicates upstream loop logic | Silent divergence when rsl-rl-lib is bumped | Exact pin already exists. A test asserts that `MpopiPpo` with `replay_ratio=0` gives bit-identical parameters to `PPO.update` from the same seed and state. |
| Stale time-out bootstrap baked into rewards | Biased replay returns | Store raw rewards and `time_outs` and re-bootstrap (§2). |
| State-distribution mismatch `d^μ ≠ d^{π_old}` | Uncorrected bias (only action-level IS is tractable) | Bound policy age. Document it as an approximation. Log policy-version distance and `KL(π_old‖μ)`. |
| Heavy-tailed IS ratios | Gradient variance and instability | Clamp log ratios, truncate, ESS gate, age filter, full diagnostics. |
| Recurrent policies | Replay would need stored hidden states and burn-in | **Unsupported in v1.** Raise on `is_recurrent`. |
| Symmetry augmentation and RND | `augment_batch` and RND normalisers are unaware of replay fields | **Unsupported in v1.** Raise if configured. |
| Multi-GPU | Each rank has its own buffer and makes local gating decisions | Gradient all-reduce happens once per minibatch, so every rank must run the same number of minibatch steps. The replay slice has a **fixed size**, and rejected samples get weight 0 instead of being dropped. That keeps minibatch counts and tensor shapes static on every rank. Tested single-process only. |
| GPU memory | `K·T·N` samples of obs plus actor/critic forward per iteration | `replay_buffer_size` counts segments. Forward passes are chunked. G1 at N=4096, T=24, K=4 is about 393k samples, which is small next to the sim. |
| Checkpoint/resume | The replay buffer is not checkpointed | Document it. On resume the buffer starts empty, which is correct but temporarily equals PPO. |
| `agent.yaml` gains an `mpopi:` section in PPO mode | Cosmetic change to dumped params | Acceptable. Not passed to PPO. |

## 6. Assumptions

- Behavior policies are **past PPO actors** of this same run
  (`μ_j = π_{θ_j}`, j < k). Their exact per-sample `log μ` and distribution
  params are recorded at act time. External or demonstration data (unknown μ) is
  out of scope. It would need estimated behavior densities.
- Actor distributions are those shipped by RSL-RL 5.5.1 (Gaussian,
  heteroscedastic Gaussian, Beta). All expose `log_prob` and `params`.
- Feed-forward actor and critic only (see Risks).
- The default `auto_reset=True` semantics (§2) apply.

## 7. Naming ambiguity (needs your confirmation)

"MPOPI" is not a standard algorithm name I can map to a published method. It
resembles **MPO** (Abdolmaleki et al. 2018) and **MPPI** (model-predictive path
integral). Neither is an importance-sampling correction layer:

- MPO's E-step does produce *reweighted* samples, `q(a|s) ∝ π_old(a|s)·exp(Q(s,a)/η)`,
  which could also be fed to PPO. Its weights are exponentiated advantages under
  a KL dual, not `π/μ`. That is a different objective.
- MPPI is a sampling-based trajectory optimiser and needs a dynamics model.

Your specification defines MPOPI operationally, as a correction layer using
`ρ = π/μ`. The design in `docs/mpopi_design.md` follows that definition. If you
meant MPO-style E-step weighting, the estimator in §3 of the design changes, but
the storage, integration and metrics plumbing does not.
