.. _recorders:

Recorders
=========

The recorder manager provides lifecycle hooks for logging data during
rollouts. Unlike rewards, recorders have no effect on the optimization
loop. They exist purely to let you capture observations, actions, or any
other environment state without modifying mjlab internals.

Each recorder term is a class that you implement. mjlab calls its methods
at the right moments and leaves all I/O decisions to you. If the ``recorders``
dictionary on ``ManagerBasedRlEnvCfg`` is empty, the environment substitutes a
lightweight no-op manager with zero overhead.


Lifecycle hooks
---------------

The manager exposes three hooks per term:

``record_pre_reset(env_ids)``
    Called inside ``env.step()`` before terminated environments are reset.
    ``obs_buf`` holds the observation from the end of the *previous* step
    (the input the agent used to choose the terminal action, not the
    post-action terminal state). ``action_manager.action`` holds the
    terminal action and is still valid here; it will be zeroed for these
    environments by ``_reset_idx`` immediately after. ``reward_buf`` holds
    the terminal reward. This is the right place to record the terminal
    transition ``(obs_t, action_t, reward_t, done=True)``.

``record_post_reset(env_ids)``
    Called after a reset completes and fresh observations are available.
    Fires at the end of ``env.reset()`` (all environments) and within
    ``env.step()`` after each batch of terminated environments is reset.
    ``obs_buf[env_ids]`` holds the initial observation of the new episode;
    ``action_manager.action[env_ids]`` is zero. Use this to initialize
    per-episode state or record the first observation.

``record_post_step()``
    Called at the end of every ``env.step()`` with fresh observations.
    For environments that reset during this step, ``action_manager.action``
    has been zeroed and ``obs_buf`` holds the initial state of the new
    episode rather than the post-action terminal observation. Use
    ``record_pre_reset`` for those environments' terminal transitions and
    ``self._env.reset_buf`` to identify which environments reset.

``close()``
    Called when the environment closes. Release file handles and flush
    buffers here.


Writing a recorder term
------------------------

Subclass :class:`~mjlab.managers.RecorderTerm` and override whichever
hooks you need. The environment is available as ``self._env``, giving
access to ``self._env.obs_buf``, ``self._env.action_manager.action``, and
all other managers.

.. code-block:: python

    import csv
    from mjlab.managers import RecorderTerm, RecorderTermCfg

    class CsvRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._file = open(cfg.params["path"], "w", newline="")
            self._writer = csv.writer(self._file)

        def record_pre_reset(self, env_ids):
            # Terminal transition: action is still intact here.
            # It will be zeroed by _reset_idx immediately after this returns.
            obs = self._env.obs_buf["actor"][env_ids].cpu().numpy()
            act = self._env.action_manager.action[env_ids].cpu().numpy()
            for o, a in zip(obs, act):
                self._writer.writerow(o.tolist() + a.tolist())

        def record_post_step(self):
            # Skip envs that just reset: their terminal pair was written
            # in record_pre_reset and their action is now zeroed.
            mask = ~self._env.reset_buf
            obs = self._env.obs_buf["actor"][mask].cpu().numpy()
            act = self._env.action_manager.action[mask].cpu().numpy()
            for o, a in zip(obs, act):
                self._writer.writerow(o.tolist() + a.tolist())

        def close(self):
            self._file.close()

The term receives the full ``cfg`` object so it can read any values you
put in ``cfg.params``.


Registration
------------

Add the term to the ``recorders`` dictionary on your environment config:

.. code-block:: python

    from dataclasses import dataclass, field
    from mjlab.managers import RecorderTermCfg

    @dataclass
    class MyEnvCfg(SomeTaskEnvCfg):
        recorders: dict = field(default_factory=lambda: {
            "csv": RecorderTermCfg(
                func=CsvRecorder,
                params={"path": "rollout.csv"},
            )
        })

Multiple terms can be registered under different keys and run together.

.. note::

    ``func`` must be a :class:`~mjlab.managers.RecorderTerm` subclass.
    Function-based terms are not supported because recorder terms are
    stateful.
