============================
Teacher-Student Distillation
============================

mjlab supports DAgger-style teacher-student distillation and subsequent RL
fine-tuning of the distilled policy, following the recipe of `Parkour in the
Wild <https://arxiv.org/abs/2505.11164>`_ (Rudin et al., 2025). The typical
use case: a teacher trained with privileged observations (e.g. a terrain
height scan) is compressed into a student that only sees deployable
observations (e.g. proprioception only), and the student is then fine-tuned
with PPO to recover or exceed teacher performance.

The pipeline has three stages:

1. **Train a teacher** with RL as usual (e.g.
   ``Mjlab-Velocity-Rough-Unitree-G1``).
2. **Distill** the teacher into a student with
   ``MjlabDistillationRunner``. The student acts in the environment while
   the frozen teacher labels every visited state, and the student
   regresses onto those labels. Because data is collected under the
   student's own state distribution (with zero-mean Gaussian exploration
   noise from its action distribution), this is on-policy dataset
   aggregation (DAgger), not behavior cloning.
3. **RL fine-tune** the student with PPO, keeping the privileged critic
   observations (asymmetric PPO). The actor is initialized from the
   distillation checkpoint via ``init_checkpoint``, starts with a reduced
   action std, and is frozen for an initial critic warmup phase so the
   randomly initialized value function cannot destroy the distilled
   behavior.

Example: G1 velocity
--------------------

Stage 1 -- train the rough-terrain teacher:

.. code-block:: sh

   uv run train Mjlab-Velocity-Rough-Unitree-G1

Stage 2 -- distill into a blind recurrent student. The student sees the
teacher's proprioceptive observations (with noise) but not the height scan,
so an LSTM is used to infer the terrain from history:

.. code-block:: sh

   uv run train Mjlab-Velocity-Rough-Unitree-G1-Distill \
     --agent.teacher-checkpoints /path/to/teacher/model_30000.pt

Stage 3 -- RL fine-tune the distilled student:

.. code-block:: sh

   uv run train Mjlab-Velocity-Rough-Unitree-G1-Distill-Finetune \
     --agent.init-checkpoint /path/to/distill/model_5000.pt

Configuration
-------------

Distillation is configured with ``RslRlDistillationRunnerCfg``:

- ``student`` / ``teacher``: model configs. The teacher config must match
  the architecture the teacher checkpoint was trained with (including its
  distribution config) so its ``actor_state_dict`` loads strictly. Give
  the student a Gaussian distribution: its ``init_std`` sets the fixed
  exploration-noise scale during data collection (the std receives no
  gradient from the distillation loss).
- ``teacher_checkpoints``: checkpoint path(s) for the frozen teacher(s).
- ``obs_groups``: maps the ``student`` and ``teacher`` observation sets to
  environment observation groups. The environment must define these
  groups; see ``mjlab.tasks.velocity.distillation_env_cfg`` for helpers
  that derive them from an existing actor/critic layout.
- ``inherit_env_state_from_teacher``: restores the environment's
  ``common_step_counter`` from the teacher checkpoint so time-based
  curricula and randomization schedules start fully ramped up.

Fine-tuning uses the regular ``RslRlOnPolicyRunnerCfg`` with two additions:

- ``init_checkpoint``: initializes the actor from a distillation (or PPO)
  checkpoint. Only model weights are restored; distribution parameters are
  dropped so the configured (reduced) ``init_std`` applies.
- ``RslRlCriticWarmupPpoAlgorithmCfg.critic_warmup_updates``: number of
  PPO updates during which the actor stays frozen while the critic trains.

Multi-expert distillation
-------------------------

Several skill- or terrain-specific experts can be distilled into a single
generalist student in one run. Configure the teacher as a
``RslRlMultiTeacherModelCfg`` with one model config per expert and one
checkpoint per expert in ``teacher_checkpoints``. Each environment is
labeled by the expert selected through an integer observation group
(``assignment_group``, default ``"teacher_assignment"``), e.g. the per-env
terrain type exposed by ``mjlab.tasks.velocity.mdp.terrain_type``. Pass
``teacher_assignment=True`` to
``mjlab.tasks.velocity.distillation_env_cfg.to_distillation_env_cfg`` to
add that group to a velocity task.

Iterative extension (new skills without forgetting old ones) works by
chaining stages: use ``init_checkpoint`` to seed a new distillation round
with the previous generalist student, add the new expert to the teacher
set, and fine-tune again on the expanded task distribution.
