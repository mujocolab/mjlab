# MPOPI → PPO: Mathematical and Software Design

Status: Phase 4 deliverable (design only, nothing implemented). It builds on
`docs/mpopi_architecture_audit.md`. Section numbers in square brackets, such as
[A§2], refer to that audit.

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| k | Current PPO iteration |
| θ_k, π_old | Actor parameters at the start of update k; `π_old = π_{θ_k}`. This policy collected the fresh rollout. |
| π_θ | Actor being optimised during update k |
| μ_j = π_{θ_j}, j < k | Behavior policy that collected replay segment j |
| age | k − j ≥ 1 |
| V = V_{φ_k} | Critic at the start of update k |
| ρ_t | π_old(a_t\|s_t) / μ(a_t\|s_t), computed as `exp(log π_old − log μ)` |
| ρ̄, c̄ | Truncation levels for the surrogate weight and for the traces |
| r_t(θ) | π_θ(a_t\|s_t) / π_old(a_t\|s_t), PPO's proximal ratio |
| ε | PPO `clip_param` |
| d^π | Discounted state-visitation distribution of π |

## 2. Why PPO needs on-policy data

PPO maximises a first-order surrogate of the improvement `J(π_θ) − J(π_old)`:

```
L_{π_old}(θ) = E_{s∼d^{π_old}} E_{a∼π_old} [ r_t(θ) · A^{π_old}(s,a) ]
```

Three things in this expression are tied to π_old:

1. **Actions** are drawn from π_old. That is why `r_t` has π_old in its
   denominator.
2. **States** are drawn from `d^{π_old}`. The TRPO/CPI lower bound
   `J(π_θ) ≥ J(π_old) + L − C·max_s KL` is derived for this state distribution.
3. **Advantages** are those of π_old, estimated by GAE on π_old's own
   trajectories.

The clip `clip(r, 1−ε, 1+ε)` is a trust region *around π_old*, and the
adaptive LR schedule measures `KL(π_old ‖ π_θ)` (`ppo.py:241-266`).

Replay samples violate all three: actions come from μ, states from `d^μ`,
and GAE on μ's trajectories estimates `A^μ`, not `A^{π_old}`.

## 3. Candidate objectives for replay data

### 3.1 The identity

For any (s, a) where the ratios are defined:

```
π_θ/μ  =  (π_θ/π_old) · (π_old/μ)  =  r_t(θ) · ρ_t
```

This is algebra and holds exactly. **Unclipped, every factorisation gives the
same objective and the same gradient.** The design choice is only about
*which quantity is clipped or truncated*.

### 3.2 Option A: μ-anchored PPO (clip the product). Rejected.

```
ℓ_A = min( (π_θ/μ)·Â,  clip(π_θ/μ, 1−ε, 1+ε)·Â )
```

At θ = θ_k the clipped ratio evaluates to `ρ_t`, not 1:

- If `Â > 0` and `ρ_t > 1+ε`, the min selects the constant branch, so the
  gradient is **zero**. The sample carries no signal even though π_old already
  favours the action.
- If `Â < 0` and `ρ_t > 1+ε`, the unclipped branch is selected. The gradient is
  `ρ_t·Â·∇log π_θ`, which is **unbounded in ρ_t**.
- The trust region is centred on μ, a policy that is no longer being updated.
  After several iterations of drift, π_old itself may lie outside it.

This is a different objective from PPO. It is also what happens if stored
`log μ` is fed into RSL-RL's unmodified `old_actions_log_prob`.

### 3.3 Option B: decoupled clipped surrogate (clip the proximal ratio, weight by ρ). Chosen.

```
ℓ_MPOPI = w̄_t · min( r_t(θ)·Â_t,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ),
w̄_t = clip(ρ_t, ρ_min, ρ̄)    (a constant with respect to θ; no gradient)
```

This is the *decoupled* PPO objective of Hilton et al. (2021, "Batch
size-invariance for policy optimization"), with π_old as the proximal policy.
GePPO (Queeney et al., 2021) is closely related: it reuses samples from recent
policies with a trust region around the current policy.

**Why it is consistent.** Assume no truncation (`w̄ = ρ`) and support
`supp π_old ⊆ supp μ`. Then for any function f:

```
E_{a∼μ}[ (π_old/μ) · f(a) ]  =  E_{a∼π_old}[ f(a) ]
```

Therefore:

```
E_{s∼d^μ, a∼μ}[ ℓ_MPOPI ]  =  E_{s∼d^μ} E_{a∼π_old}[ min(r Â, clip(r) Â) ]
```

That is **exactly PPO's clipped surrogate, evaluated on state distribution
`d^μ` instead of `d^{π_old}`.** The trust region stays around π_old, and at
θ = θ_k every sample has `r = 1`, as in PPO.

**Gradient check (used as a unit test).** At θ = θ_k, clipping is inactive and:

```
∇θ ℓ_MPOPI = w̄_t · Â_t · ∇θ log π_θ(a_t|s_t)
```

This is the truncated-IS policy gradient. Without truncation it equals
`(π_θ/μ)·Â·∇log π_θ`, which is also Option A's unclipped gradient. That
confirms the identity in §3.1.

### 3.4 What remains uncorrected: state distribution

Correcting `d^μ → d^{π_old}` requires products of ratios along whole
trajectories or a learned density ratio. Both have very high variance, and v1
does not attempt them. **The MPOPI objective is therefore a biased surrogate.**
Its bias is controlled by how far each μ_j is from π_old. GePPO derives a
generalised policy-improvement bound whose penalty grows with the
total-variation distance between π_old and the behavior policies, and that
distance grows with policy age.

This is the justification for `max_policy_age` and for logging
`KL(μ ‖ π_old)`. It is an approximation, not an exact correction, and the
documentation will say so.

## 4. Advantage and value estimator (V-trace / truncated-IS GAE)

GAE on a μ-trajectory with the current critic estimates advantages of μ in
its multi-step part. MPOPI therefore recomputes targets for every replay
segment at the start of each update, with V = V_{φ_k} and ρ from π_old.

Per segment of length T (t = T−1 … 0):

```
r̃_t = r_t^raw + γ · V(s_t) · timeout_t              (re-applies RSL-RL's time-out
                                                      bootstrap with the CURRENT
                                                      critic, [A§2])
δ_t = r̃_t + γ (1−d_t) V(s_{t+1}) − V(s_t)           V(s_T) = V(bootstrap obs)
ρ̄_t = min(ρ̄, ρ_t),   c_t = min(c̄, ρ_t)
v_t = V(s_t) + ρ̄_t δ_t + γλ (1−d_t) c_t (v_{t+1} − V(s_{t+1})),   v_T = V(s_T)
Â_t = δ_t + γλ (1−d_t) (v_{t+1} − V(s_{t+1}))
```

- `v_t` is the V-trace target (Espeholt et al., 2018), with the per-step trace
  coefficient `λ·c_t`. It is the critic's regression target (`returns`).
- `Â_t` is the advantage used in the surrogate. It does *not* contain `ρ_t`,
  because `a_t` is conditioned on. The correction for `a_t` is the surrogate
  weight `w̄_t`. Later actions are corrected inside `v_{t+1}`.
- λ = 0 gives the 1-step TD advantage. λ = 1 gives IMPALA's
  `r + γ v_{t+1} − V(s_t)`.

**Reduction to PPO (tested).** When ρ ≡ 1 and ρ̄, c̄ ≥ 1:

```
v_t − V(s_t) = δ_t + γλ(1−d_t)(v_{t+1} − V(s_{t+1}))
```

This is exactly RSL-RL's GAE recursion (`ppo.py:176-186`), and `Â_t` equals GAE.
The fresh rollout still uses upstream `compute_returns` unchanged. Test 9c checks
that the V-trace code with ρ ≡ 1 matches it to floating-point tolerance.

**Fixed point and bias.** With truncation active, V-trace converges to
`V^{π_ρ̄}`, where `π_ρ̄(a|s) ∝ min(ρ̄·μ(a|s), π_old(a|s))`. That policy lies
between μ and π_old, so the advantages are biased toward μ. With ρ̄ = ∞ there is
no truncation bias, at the cost of heavy-tailed variance. V-trace assumes
`ρ̄ ≥ c̄`, and the config validator warns otherwise. c̄ ≤ 1 keeps the trace
product bounded (Retrace-style contraction).

**Value loss.** Unweighted over accepted samples. The target `v_t` already
contains the action correction. The state distribution for regression is the
fresh/replay mixture, which is fine for a baseline. The PPO value-clip anchor
for replay is `V_{φ_k}(s_t)` (recomputed), which matches the meaning of
`batch.values` for fresh data.

## 5. Importance-weight pipeline

Everything is done in the log domain, per replay sample:

| Step | Operation | Purpose | Bias / variance effect |
|------|-----------|---------|------------------------|
| 1 | `log ρ = log π_old(a\|s) − log μ(a\|s)` | Ratio without dividing probabilities | Exact |
| 2 | Reject if either log-prob is non-finite | Beta at a boundary, corrupted data | Drops samples (selection bias ≈ 0 for Gaussian) |
| 3 | Reject if `age > max_policy_age` (segment level) | Stale policy | Bias ↓ (state shift), variance ↑ (fewer samples) |
| 4 | Reject if `\|log ρ\| > max_abs_log_ratio` (optional) | Coverage/support proxy: π_old puts mass where μ had almost none | Selection bias; removes extreme tails |
| 5 | `log ρ ← clamp(log ρ, −L, L)`, L = `log_ratio_clamp` | **Numerical only.** L = 20 gives exp(20) ≈ 4.9e8, finite in fp32 and bf16. The clamp should be inactive once steps 4 and 6 are set. | None when inactive |
| 6 | `w̄ = clip(exp(log ρ), ρ_min, ρ̄)` | Truncation (ρ̄) and optional lower clip (ρ_min, default 0) | ρ̄: bias toward μ, variance bounded by ρ̄². ρ_min > 0 over-weights actions π_old dislikes. |
| 7 | ESS gate: `ESS_n = (Σw̄)² / (n·Σw̄²)` over accepted replay. If `< min_ess`, zero out all replay this iteration. | Detect a degenerate weight distribution | Falls back to exact PPO for that iteration |
| 8 | Optional self-normalisation: `w̄ ← w̄ / mean(w̄)` over accepted replay | Keeps replay's total weight fixed | Biased but consistent. Scale-invariant. |

Rejected samples are **not removed**. They get `mask = 0`, so tensor shapes and
minibatch counts stay static. That matters for multi-GPU [A§5] and for
compiled models. The fresh rollout always has `w = 1` and `mask = 1`.

**Zero behavior probability.** For RSL-RL's Gaussian `μ > 0` everywhere, so the
support condition holds analytically. In floating point, `log μ` can be very
negative, which gives a huge `log ρ`. Steps 4–6 handle that. For Beta, `log μ =
−∞` at the boundary. Step 2 handles it.

## 6. The full MPOPI objective (a new objective, not standard PPO)

Let F be the fresh samples (w = 1) and R the replay samples (weights w̄,
masks m). Let n = |F| + Σ_R m_i. Then:

```
L(θ, φ) =  (1/n) Σ_{i∈F∪R} m_i [ −w_i·min(r_i Â_i, clip(r_i,1±ε) Â_i)
                                 + c_v · ValueLossClipped(V_φ(s_i), v_i, V_{φ_k}(s_i))
                                 − c_e · H[π_θ(·|s_i)] ]
```

- Advantages are normalised over accepted samples, unweighted, mirroring
  upstream's batch normalisation. The rule is identical in modes B and C, so it
  cannot confound them.
- It uses the same optimizer, the same `num_learning_epochs ×
  num_mini_batches` optimizer steps, grad clipping, and adaptive-LR rule on
  `KL(π_old‖π_θ)`. For replay samples the old distribution params are π_old's,
  recomputed. Only the minibatch *size* grows by `(1 + replay_ratio)`.
- Effective state distribution: a mixture of `d^{π_old}` and the `d^{μ_j}`.

**There is one optimizer.** MPOPI has no parameters and never calls
`optimizer.step()`.

### 6.1 Estimator summary

| Component | Exact? | Bias source | Variance source |
|-----------|--------|-------------|-----------------|
| Action correction (ρ̄ = ∞) | Unbiased given support | none | ρ² tails |
| Action correction (ρ̄ < ∞) | Biased | `min(ρ, ρ̄)` targets π_ρ̄ | Bounded by ρ̄² |
| Multi-step targets (V-trace) | Biased if truncated | Fixed point `V^{π_ρ̄}`, critic error | Trace products, bounded by c̄ ≤ 1 |
| State distribution | **Not corrected** | `d^μ ≠ d^{π_old}` grows with age | — |
| Self-normalisation | Biased, consistent | Ratio estimator | Reduced |
| Sample rejection | Selection bias | Drops high-ρ tails | Reduced |

## 7. Experimental modes

| Mode | Replay used | Surrogate weight | Traces | PPO anchor for replay |
|------|-------------|------------------|--------|-----------------------|
| A `ppo` | no | — | — | — (upstream `PPO` class, untouched) |
| B `naive_replay_ppo` | yes | 1 | λ (plain GAE with current critic) | recomputed π_old |
| C `mpopi_ppo` | yes | w̄ | λ·c_t | recomputed π_old |

B and C share storage, sampling, rejection masks, recomputed critic,
re-bootstrapping, advantage normalisation and minibatching. **The only
difference is ρ.** A difference between B and C is therefore attributable to
the importance correction itself.

B "labels replay as on-policy" deliberately, and it is reported as such. A
possible fourth arm, D = μ-anchored PPO (Option A, §3.2), is left for later,
because it changes two things at once relative to B.

## 8. Data structures

### 8.1 `ReplayBuffer` (GPU ring over rollout *segments*)

A segment is one full `[T, N]` rollout. Segments are stored rather than single
transitions because V-trace needs contiguous traces [A§2]. Capacity is
`replay_buffer_size` segments.

| Field | Shape | Notes |
|-------|-------|-------|
| `observations` | TensorDict `[K, T, N]` | Same groups and dtypes as `RolloutStorage` |
| `actions` | `[K, T, N, A]` | Unclipped samples [A§2] |
| `rewards` | `[K, T, N, 1]` | **Raw**, captured before RSL-RL's time-out bootstrap |
| `dones` | `[K, T, N, 1]` | terminated ∨ truncated |
| `time_outs` | `[K, T, N, 1]` | truncated. terminated = dones ∧ ¬time_outs |
| `behavior_log_prob` | `[K, T, N, 1]` | log μ(a\|s), taken from `storage.actions_log_prob` |
| `behavior_dist_params` | tuple of `[K, T, N, ·]` | For `KL(μ‖π_old)` diagnostics |
| `bootstrap_observations` | TensorDict `[K, N]` | s_T for the segment |
| `policy_version` | `[K]` int64 | Iteration j that produced the segment |
| `occupied` | `[K]` bool | |

There is deliberately **no per-step `next_observation`**. Within a segment,
`s_{t+1}` is `observations[t+1]` (or the bootstrap obs). On done steps a true
next state does not exist because of auto-reset [A§2]. Episode and timestep
metadata beyond `dones` is not needed by the estimator. Priorities are deferred:
prioritised sampling adds its own IS correction, which is a second bias layer
to design separately.

Insertion copies the fresh `RolloutStorage` tensors at the end of `update()`
(GPU→GPU). Memory per segment equals one `RolloutStorage`.

### 8.2 `Mpopi.process` → `MpopiBatch`

```python
class Mpopi:
  def process(self, buffer, actor, critic, version) -> MpopiBatch: ...

@dataclass
class MpopiBatch:            # flat [n_replay] tensors
  observations: TensorDict
  actions: torch.Tensor
  values: torch.Tensor                   # V_{φ_k}(s)
  advantages: torch.Tensor               # Â (unnormalised)
  returns: torch.Tensor                  # v (V-trace)
  old_actions_log_prob: torch.Tensor     # log π_old(a|s)
  old_distribution_params: tuple[torch.Tensor, ...]   # π_old
  behavior_actions_log_prob: torch.Tensor  # log μ(a|s)
  weights: torch.Tensor                  # w̄ (1 in naive mode)
  mask: torch.Tensor                     # accepted
  policy_age: torch.Tensor
  metrics: dict[str, float]
```

The field names mirror `RolloutStorage.Batch`, so the minibatch generator
concatenates the fresh and replay pools and yields upstream `Batch` objects
plus `(weights, mask)`. `current_policy` in your sketch is π_old (the actor at
θ_k). `process` runs under `torch.inference_mode()`, chunked over segments.

## 9. Configuration (`MpopiCfg`, nested at `RslRlPpoAlgorithmCfg.mpopi`)

| Field | Default | Rationale |
|-------|---------|-----------|
| `mode` | `"ppo"` | `ppo` / `naive_replay_ppo` / `mpopi_ppo`. Replaces `enable_mpopi` (one source of truth; `enable_mpopi ≡ mode != "ppo"`). |
| `replay_buffer_size` | 4 | Segments. **Placeholder, no empirical basis yet.** Ablated. |
| `replay_ratio` | 1.0 | Replay samples per fresh sample. |
| `replay_batch_size` | None | If set, overrides `replay_ratio` (absolute count). |
| `sampling_strategy` | `"uniform"` | `uniform` over eligible transitions, or `all` (use every eligible transition, ignoring the ratio). |
| `max_policy_age` | None | Segment-level stale filter. None means up to buffer capacity. |
| `max_abs_log_ratio` | None | Sample-level support filter (§5 step 4). |
| `importance_weight_clip_min` | 0.0 | ρ_min. 0 means no lower clip. |
| `importance_weight_clip_max` | 1.0 | ρ̄. 1.0 follows V-trace/IMPALA. None means no truncation (ablation 4). |
| `trace_clip_max` | 1.0 | c̄. ≤ 1 for bounded trace products. |
| `log_ratio_clamp` | 20.0 | Numerical guard only (§5 step 5). |
| `min_ess` | 0.0 | Normalised ESS gate in [0, 1]. 0 disables. |
| `weight_normalization` | `"none"` | `none` / `self_normalized` |

`stale_data_filter` from your list is implemented by `max_policy_age`
(segments) and `max_abs_log_ratio` (samples). `importance_sampling` on/off is
the `mode` (B vs C). No experimental constant is hard-coded. The only defaults
with a literature basis are ρ̄ = c̄ = 1. Every other default either disables its
feature or is marked as a placeholder.

CLI (planned):

```
uv run train Mjlab-Cartpole-Balance --agent.algorithm.mpopi.mode mpopi_ppo \
  --agent.algorithm.mpopi.replay-ratio 1.0 --agent.algorithm.mpopi.replay-buffer-size 4
```

## 10. Per-iteration schedule (inside `MpopiPpo`)

```
rollout (upstream, unchanged)   act → env.step → process_env_step
                                 └─ MpopiPpo also records raw rewards + time_outs
compute_returns (upstream GAE on fresh) + remember bootstrap obs
update():
  1. batch_R = Mpopi.process(buffer, actor=π_old, critic=V_{φ_k}, version=k)
       eligible segments → forward π_old, V → log ρ → filters → w̄
       → V-trace v, Â → sample replay_batch_size → ESS gate
  2. pool = fresh (w=1, m=1) ⊕ batch_R; normalise Â over accepted samples
  3. for epoch, minibatch in pool: weighted PPO loss → backward → clip → step
  4. update obs normalisers on FRESH obs only (upstream behaviour)
  5. buffer.insert(fresh segment, version=k)   → replay ages are always ≥ 1
  6. storage.clear(); return loss_dict (+ kl, clip_fraction, mpopi/*)
```

## 11. Metrics (all through `loss_dict` and logged as `Loss/<key>`)

- **PPO:** `surrogate`, `value`, `entropy` (existing), plus new `kl`
  (mean `KL(π_old‖π_θ)` over minibatches) and `clip_fraction` (fraction with
  `|r−1| > ε`, accepted samples). Episode reward and length come from the
  existing `Train/*`.
- **MPOPI:** `mpopi/buffer_segments`, `buffer_transitions`, `sampled`,
  `accepted`, `rejected_{nonfinite,age,log_ratio,ess}`, `raw_ratio_{mean,std,
  min,max}`, `weight_{mean,std,min,max}`, `clipped_frac`, `ess`,
  `policy_age_{mean,max}`, `stale_frac`, `behavior_kl` (mean `KL(μ‖π_old)`,
  the policy-version distance in distribution space), and `gradient_samples`
  (fresh + accepted).
- **Sample efficiency:** env steps come from the logger's existing total steps
  (unchanged by replay). Reward vs env steps is `Train/mean_reward` × the
  constant `N·T` per iteration. Updates per env step come from
  `gradient_samples` and the constant optimizer steps per iteration.

## 12. Implementation plan (files)

New: `src/mjlab/rl/mpopi/{__init__,config,estimators,replay_buffer,mpopi,algorithm}.py`.
`estimators.py` holds pure functions (log ratio, truncation, ESS, V-trace),
which keeps them trivially testable on CPU.

Modified: `src/mjlab/rl/config.py` (the `mpopi` field),
`src/mjlab/rl/runner.py` (mode translation, §A4.3), and
`src/mjlab/rl/__init__.py` (exports).

Docs: `docs/source/training/mpopi.rst` plus a toctree entry, and a changelog
entry. Removal means deleting `rl/mpopi/`, the config field, and one block in
the runner.

## 13. Test plan

| # | Test | Setup |
|---|------|-------|
| 1 | Ratio | log(0.8) − log(0.4) → ρ = 2.0 |
| 2 | Truncation and lower clip | Known inputs to `clip_weights` |
| 3 | Near-zero μ | `log μ = −1e4`, `-inf` → finite output, rejected or masked |
| 4 | Log-domain stability | log-probs ±1e3 → no overflow; matches float64 reference |
| 5 | ESS | Uniform weights → 1. One-hot → 1/n. |
| 6 | Buffer | Insert past capacity → FIFO eviction, versions, shapes, device |
| 7 | Version filter | `max_policy_age` masks the right segments |
| 8 | Batch generation | `process` on a hand-built buffer → shapes, masks, weights |
| 9 | PPO unchanged | (a) `mode="ppo"` → runner builds upstream `PPO` with today's kwargs. (b) `MpopiPpo` with an empty buffer → parameters bit-identical to `PPO.update` from the same seed. (c) V-trace with ρ ≡ 1 equals upstream GAE. |
| 10 | Known correction direction | Bandit: μ = N(0,1), π_old = N(1,1), reward = a. The weighted estimate of E_{π_old}[r] ≈ 1, while naive gives ≈ 0. The weighted surrogate gradient on the mean points toward +. |
| 11 | Gradient | At θ_k, ∇ℓ_MPOPI equals `w̄·Â·∇log π` (autograd vs analytic) |
| 12 | Extreme ratios | log ρ ∈ {±1e4, nan, inf} → loss and grads finite |

Toy validation: a pure-torch `rsl_rl.env.VecEnv` (1-D point mass, reward
−x², 64 envs, CPU, seconds per run). Modes A/B/C × 5 seeds, reward vs env
steps. Only after that, `Mjlab-Cartpole-Balance`, then a velocity task.

## 14. Ablation matrix

| ID | mode | ρ̄ | replay_ratio | buffer | max_age | Question |
|----|------|----|--------------|--------|---------|----------|
| 1 | ppo | — | 0 | — | — | Baseline |
| 2 | naive_replay_ppo | — | 1 | 4 | — | Does uncorrected replay help or hurt? |
| 3 | mpopi_ppo | 1 | 1 | 4 | — | Does the correction matter (vs 2)? |
| 4 | mpopi_ppo | None | 1 | 4 | — | Truncation bias vs variance |
| 5 | mpopi_ppo | {0.5, 1, 2} | 1 | 4 | — | Truncation level |
| 6 | mpopi_ppo | 1 | 1 | 8 | {1, 2, 4, 8} | Staleness |
| 7 | mpopi_ppo | 1 | {0.25, 0.5, 1, 2} | 4 | — | Replay ratio |

Track reward vs env steps, seed variance, `kl`, `ess`, `weight_std`,
`behavior_kl`, and `clip_fraction`. At least 5 seeds per cell. Report
confidence intervals. **Make no claim that MPOPI helps until 3 vs 1 and 3 vs 2
are significant.**

## 15. Known limitations (v1)

Feed-forward policies only. No symmetry or RND. The state-distribution shift is
uncorrected (§3.4). The replay buffer is not checkpointed. Multi-GPU is
untested. The metric prefix is `Loss/`. `update()` mirrors upstream 5.5.1 and
must be re-checked on any rsl-rl-lib bump (test 9b guards this).

## References

- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*.
- Espeholt et al. 2018, *IMPALA* (V-trace).
- Munos et al. 2016, *Safe and Efficient Off-Policy RL* (Retrace).
- Hilton, Cobbe, Schulman 2021, *Batch Size-Invariance for Policy Optimization* (decoupled objective).
- Queeney, Paschalidis, Cassandras 2021, *Generalized Proximal Policy Optimization with Sample Reuse* (GePPO).
