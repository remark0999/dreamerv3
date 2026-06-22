# COMP9991 DMC Environment Seeding Plan

Scope: planning only. This document audits deterministic seeding for the DMC visual walker-walk environment path and proposes a later implementation pass. It does not modify source code, configs, dependencies, tests, or training behavior.

Inputs read:

- `docs/comp9991/reference_audit.md`
- `docs/comp9991/noise_wrapper_plan.md`
- Local source in `dreamerv3/main.py` and `embodied/envs/dmc.py`
- dm_control 1.0.41 source:
  - `dm_control/suite/walker.py`: https://raw.githubusercontent.com/google-deepmind/dm_control/1.0.41/dm_control/suite/walker.py
  - `dm_control/suite/__init__.py`: https://raw.githubusercontent.com/google-deepmind/dm_control/1.0.41/dm_control/suite/__init__.py

## Facts Found In Code

### Current seed path in `dreamerv3/main.py`

- `main()` loads `dreamerv3/configs.yaml`, starts from `defaults`, applies named configs from `--configs`, parses CLI flags, then saves the resolved config to `logdir/config.yaml` before creating envs (`dreamerv3/main.py:26`-`dreamerv3/main.py:45`).
- `defaults.seed` is `0` (`dreamerv3/configs.yaml:8`).
- `make_env(config, index, **overrides)` receives the environment index from the run loop (`dreamerv3/main.py:215`).
- `make_env()` splits `config.task` into `suite, task`; for `dmc_walker_walk`, `suite == "dmc"` and `task == "walker_walk"` (`dreamerv3/main.py:215`-`dreamerv3/main.py:216`).
- For DMC, the constructor is resolved from `'embodied.envs.dmc:DMC'` (`dreamerv3/main.py:220`-`dreamerv3/main.py:241`).
- `kwargs = config.env.get(suite, {})` loads `config.env.dmc` for DMC, then applies overrides (`dreamerv3/main.py:242`-`dreamerv3/main.py:243`).
- A constructor `seed` kwarg is currently added only if `kwargs.pop('use_seed', False)` is true. That path uses Python `hash((config.seed, index)) % (2 ** 32 - 1)` (`dreamerv3/main.py:244`-`dreamerv3/main.py:245`).
- `config.env.dmc` defaults are `{size: [64, 64], repeat: 1, proprio: True, image: True, camera: -1}` and do not include `use_seed: True` (`dreamerv3/configs.yaml:36`).
- `dmc_vision` sets `task: dmc_walker_walk` and `env.dmc.proprio: False`; it does not add `use_seed: True` (`dreamerv3/configs.yaml:192`-`dreamerv3/configs.yaml:195`).
- Therefore the pinned `dmc_vision` path does not currently pass `seed` into `embodied.envs.dmc.DMC`.
- Observation-noise seeding is already wrapper-only and independent: `OBS_NOISE_SEED_NAMESPACE = 0x0B50_5015`; `_obs_noise_seed(seed, index)` uses `np.random.SeedSequence([int(seed), int(index), OBS_NOISE_SEED_NAMESPACE])` and returns one generated `np.uint32` state as an `int` (`dreamerv3/main.py:19`, `dreamerv3/main.py:274`-`dreamerv3/main.py:277`).

### Current DMC construction in `embodied/envs/dmc.py`

- `DMC.__init__()` currently accepts `env, repeat=1, size=(64, 64), proprio=True, image=True, camera=-1`; it has no `seed` argument (`embodied/envs/dmc.py:21`-`embodied/envs/dmc.py:22`).
- If `env` is a string, it splits it into `domain, task` using the first underscore (`embodied/envs/dmc.py:25`-`embodied/envs/dmc.py:26`).
- For walker, `camera == -1` maps to camera `0` because only `quadruped` and `rodent` have domain-specific default cameras (`embodied/envs/dmc.py:16`-`embodied/envs/dmc.py:28`).
- `cup` is renamed to `ball_in_cup` (`embodied/envs/dmc.py:29`-`embodied/envs/dmc.py:30`).
- Manipulation tasks use `manipulation.load(task + '_vision')` (`embodied/envs/dmc.py:31`-`embodied/envs/dmc.py:32`).
- Rodent tasks use `getattr(basic_rodent_2020, task)()` (`embodied/envs/dmc.py:33`-`embodied/envs/dmc.py:38`).
- All other standard DMC tasks use `suite.load(domain, task)` (`embodied/envs/dmc.py:39`-`embodied/envs/dmc.py:40`).
- The current pinned standard DMC path therefore calls `suite.load(domain, task)` without `task_kwargs={"random": ...}`.
- The loaded dm_control env is wrapped by `from_dm.FromDM`, then by `embodied.wrappers.ActionRepeat` (`embodied/envs/dmc.py:41`-`embodied/envs/dmc.py:43`).

### dm_control 1.0.41 facts

- In dm_control 1.0.41, `suite.load(domain_name, task_name, task_kwargs=None, environment_kwargs=None, visualize_reward=False)` accepts `task_kwargs` and forwards them to the selected task constructor through `build_environment()`.
- In dm_control 1.0.41, `walker.walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None)` passes `random=random` into `PlanarWalker`.
- `PlanarWalker.__init__(move_speed, random=None)` documents `random` as optional and permits a `numpy.random.RandomState`, an integer seed, or `None`.
- Thus `suite.load('walker', 'walk', task_kwargs={'random': seed})` is the upstream-supported seeding route for standard walker-walk in dm_control 1.0.41.

## Recommendations

### Deterministic DMC seed derivation

Add a DMC environment seed namespace distinct from observation noise:

```python
DMC_ENV_SEED_NAMESPACE = 0xD0C0_5EED
```

Derive one wrapper-independent DMC environment seed from `config.seed`, environment index, and the fixed namespace:

```python
def _dmc_env_seed(seed, index):
  sequence = np.random.SeedSequence([
      int(seed),
      int(index),
      DMC_ENV_SEED_NAMESPACE,
  ])
  return int(sequence.generate_state(1, dtype=np.uint32)[0])
```

Properties:

- Deterministic for the same run seed and environment index.
- Different for different environment indices with high probability under `SeedSequence`.
- Different for different run seeds with high probability under `SeedSequence`.
- Returns a uint32-compatible Python `int`.
- Does not use Python `hash()`.
- Does not consume or alter observation-noise RNG derivation.
- Uses a namespace distinct from `OBS_NOISE_SEED_NAMESPACE = 0x0B50_5015`.

### Minimal implementation shape

Recommended later implementation:

- In `embodied/envs/dmc.py`, add an optional `seed=None` argument to `DMC.__init__()`.
- For standard suite tasks only, replace:

```python
env = suite.load(domain, task)
```

with:

```python
task_kwargs = {'random': int(seed)} if seed is not None else None
env = suite.load(domain, task, task_kwargs=task_kwargs)
```

- Leave manipulation behavior unchanged unless separately audited: keep `manipulation.load(task + '_vision')`.
- Leave rodent behavior unchanged unless separately audited: keep `getattr(basic_rodent_2020, task)()`.
- In `dreamerv3/main.py`, pass the derived DMC seed only for `suite == 'dmc'`. Do not route DMC through the existing generic `use_seed` path because that path uses Python `hash()`.
- Do not alter `_obs_noise_seed()`, `OBS_NOISE_SEED_NAMESPACE`, `ObservationNoise`, or any observation-noise formula.
- Do not change DMC action repeat, image rendering, observation dtype conversion, replay, agent losses, or normalizers.

Suggested main-path pseudocode:

```python
DMC_ENV_SEED_NAMESPACE = 0xD0C0_5EED

def make_env(config, index, **overrides):
  ...
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  use_seed = kwargs.pop('use_seed', False)
  if suite == 'dmc':
    kwargs['seed'] = _dmc_env_seed(config.seed, index)
  elif use_seed:
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  ...
```

This preserves existing non-DMC behavior while making DMC deterministic without Python `hash()`.

### Config and metadata recommendation

The current saved `config.yaml` records `seed`, but not the derived DMC namespace or per-environment DMC seeds. Do not write this metadata from `make_env()`: that would duplicate writes across env construction and can mix training/eval/parallel roles. Write metadata exactly once in `main()`, after logdir creation and `config.yaml` saving, and before agent or environment construction.

Scope the first metadata implementation to `config.script == "train"` only. For that scope, enumerate training environment indices with `range(config.run.envs)` and record:

- `config.seed`
- `DMC_ENV_SEED_NAMESPACE`
- derivation version
- `config.script`
- environment index to derived seed mapping

Preferred minimal metadata file:

```text
{logdir}/dmc_env_seeds.json
```

Example shape:

```json
{
  "seed": 0,
  "namespace": 3502268141,
  "derivation_version": "dmc-env-seed-v1",
  "derivation": "np.random.SeedSequence([seed, index, namespace]).generate_state(1, dtype=np.uint32)[0]",
  "script": "train",
  "envs": {"0": 123, "1": 456}
}
```

This metadata does not change training behavior; it makes the formal seeding state auditable. `train_eval` and parallel role-specific seed metadata are deferred. Do not claim that an index-only derivation distinguishes train and eval environments when both begin at index zero; role-specific derivation or metadata needs a separate audit.

## Exact Files And Functions For Later Implementation

Modify only these files in the later implementation pass:

- `dreamerv3/main.py`
  - Add `DMC_ENV_SEED_NAMESPACE`.
  - Add `_dmc_env_seed(seed, index)`.
  - In `make_env()`, pass `seed=_dmc_env_seed(config.seed, index)` for `suite == 'dmc'`.
  - Ensure DMC does not use the generic Python-hash `use_seed` branch.
  - Leave `_obs_noise_seed()` unchanged.
  - Add train-script-only run metadata emission for derived DMC env seeds in `main()`, exactly once after logdir/config creation and before agent or env construction.
- `embodied/envs/dmc.py`
  - Add `seed=None` to `DMC.__init__()`.
  - For standard `suite.load()` tasks, pass `task_kwargs={'random': int(seed)}` when `seed is not None`.
  - Leave manipulation and rodent branches unchanged.
- `tools/probe_dmc_env_seed.py`
  - Add a no-training reproducibility probe.
  - Load resolved configs in the same style as `dreamerv3.main`.
  - Build environments through `dreamerv3.main.make_env()`.
  - Create zero actions from `env.act_space`.
  - Compare exact clean trajectories.
  - Do not train an agent.
- Test file, recommended `embodied/tests/test_dmc_env_seeding.py`
  - Add deterministic seed derivation and DMC construction tests.
  - Use monkeypatching where dm_control is unavailable locally.

No changes are recommended to `dreamerv3/agent.py`, `dreamerv3/rssm.py`, replay, optimizer code, objective code, dependencies, or observation-noise formulas.

## Test Plan

### Unit and integration tests

1. Same run seed and same env index produce identical reset images
   - Build two fresh DMC env instances with the same resolved config and index.
   - `embodied.Env` does not use `env.reset()` in this path.
   - For each instance, obtain the initial observation by calling `env.step()` with `reset=True` and a zero continuous action built from `env.act_space`.
   - Compare the first rendered `image` elementwise.
   - Close both environments at the end.

2. Same run seed and same env index produce identical fixed-action trajectories
   - Build two fresh env instances with same seed and index.
   - Obtain each initial observation using `env.step()` with `reset=True` and zero continuous action.
   - Step both with the same fixed zero-action sequence for `N` steps using `reset=False`.
   - Compare images, rewards, `is_first`, `is_last`, and `is_terminal` elementwise.
   - Close all created environments at the end.
   - Do not require repeated resets of one already-advanced environment to produce the same trajectory.

3. Different env indices produce different initial trajectories
   - Build env index `0` and env index `1` with the same `config.seed`.
   - Use the same fixed zero-action sequence.
   - Assert at least one image, proprio observation when enabled, reward, or terminal flag differs over the first `N` steps.
   - Use enough steps to avoid fragile equality from coincidental visual similarity.
   - Close all created environments at the end.

4. Different run seeds produce different initial trajectories
   - Build env index `0` with two different `config.seed` values.
   - Run the same fixed zero-action sequence.
   - Assert at least one trajectory element differs over the first `N` steps.
   - Close all created environments at the end.

5. Clean mode still does not instantiate `ObservationNoise`
   - Build env with `clean`.
   - Walk wrapper chain and assert no `embodied.wrappers.ObservationNoise` instance.
   - This guards the DMC-seeding change from accidentally coupling to observation noise.

6. Observation-noise RNG remains independently derived
   - Assert `_obs_noise_seed(seed, index)` still uses `OBS_NOISE_SEED_NAMESPACE`.
   - Assert `_dmc_env_seed(seed, index) != _obs_noise_seed(seed, index)` for representative seeds and indices.
   - Assert enabling `g20` or `pink` does not alter the derived DMC seed.

7. Environment seed is saved in config/run metadata
   - Run a tiny `config.script == "train"` command with a temp logdir.
   - Assert `config.yaml` contains the run seed.
   - Assert `dmc_env_seeds.json` exists and records run seed, namespace, derivation version, script, and training environment index to seed mapping for `range(config.run.envs)`.
   - Recompute seeds in test and compare with metadata.
   - Assert metadata is not emitted from `make_env()`.
   - Defer `train_eval` and parallel role-specific metadata tests.

8. Standard DMC construction passes `task_kwargs`
   - Monkeypatch `dm_control.suite.load`.
   - Construct `DMC('walker_walk', seed=123)`.
   - Assert `suite.load('walker', 'walk', task_kwargs={'random': 123})` was called.
   - Assert manipulation and rodent branches do not receive this new `task_kwargs` path unless separately audited.

9. `DMC(seed=None)` preserves original suite-load behavior
   - Monkeypatch `dm_control.suite.load`.
   - Construct `DMC('walker_walk', seed=None)`.
   - Assert the call preserves the original behavior: `suite.load('walker', 'walk')` without a random `task_kwargs`.

### No-training reproducibility probe

Use a probe that creates environments and runs a fixed zero-action sequence without training. Recommended behavior:

- Load the same resolved config path as `dreamerv3/main.py`.
- Build environments through `dreamerv3.main.make_env()`.
- Construct two fresh env index `0` instances with `make_env(config, 0)`.
- Create zero actions from each env's `act_space`, including zero-valued continuous actions and the `reset` flag.
- For each env instance, obtain the initial observation with `env.step()` using `reset=True`.
- For `N=100` subsequent steps, feed the fixed zero-action sequence with `reset=False`.
- Record per-step:
  - image bytes or image checksum,
  - reward,
  - `is_first`,
  - `is_last`,
  - `is_terminal`.
- Assert exact equality for same seed/index.
- Repeat for env indices `0` and `1` and assert divergence over the sequence.
- Close all created environments at the end.
- Do not use `env.reset()`.
- Do not train an agent.
- Do not require repeated resets of one already-advanced environment to reproduce its initial trajectory.

Suggested command form for the future implementation:

```sh
python tools/probe_dmc_env_seed.py --configs dmc_vision clean --seed 0 --env-index 0 --steps 100 --logdir ./scratch/comp9991-dmc-seed-probe/run-a
python tools/probe_dmc_env_seed.py --configs dmc_vision clean --seed 0 --env-index 0 --steps 100 --logdir ./scratch/comp9991-dmc-seed-probe/run-b
```

The probe should compare exact environment trajectories, not learning metrics.

### Short clean training repeatability check

After the fix, run two identical short clean runs with unique logdirs:

```sh
python dreamerv3/main.py --configs dmc_vision size1m clean --task dmc_walker_walk --seed 0 --run.steps 5000 --run.envs 2 --run.train_ratio 16 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-dmc-seed-train/clean-a
python dreamerv3/main.py --configs dmc_vision size1m clean --task dmc_walker_walk --seed 0 --run.steps 5000 --run.envs 2 --run.train_ratio 16 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-dmc-seed-train/clean-b
```

Interpretation:

- Environment trajectory reproducibility should be exact under the no-training probe for the same seed/index/action sequence.
- Numerical training repeatability is weaker: JAX, GPU kernels, parallel drivers, asynchronous logging, and floating-point reduction order can prevent bitwise-identical training metrics.
- For short training runs, check that runs start, metadata matches, and broad learning/runtime metrics are repeatable within normal numerical tolerance; do not require full GPU training to be bitwise identical.

## Experiment Migration Rule

All runs produced before the DMC environment-seeding fix are pilot-only. They must not be pooled with the new formally seeded runs. Reports and plots should label the pre-fix runs separately or exclude them from seeded comparisons.

## Hard Prohibitions

The later implementation must explicitly avoid:

- Disagreement instrumentation.
- Teacher gate changes.
- Actor objective changes.
- Value objective changes.
- World-model objective changes.
- Replay objective changes.
- Observation-noise formula changes.
- Observation-noise RNG derivation changes.
- Threshold tuning.
- Dependency changes.
