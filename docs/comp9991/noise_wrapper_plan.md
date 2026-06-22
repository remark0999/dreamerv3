# COMP9991 DMC Observation Noise Wrapper Plan

Scope: planning only. This document proposes a future implementation of SheepRL-style image observation noise in this Danijar DreamerV3 repository. It does not propose disagreement instrumentation, teacher gating, loss changes, return-normalization changes, or threshold tuning.

The facts below are extracted from:

- `docs/comp9991/reference_audit.md`
- `docs/comp9991/_local/sheep_noise_source_dump.txt`

## SheepRL Facts

### Wrapper location in SheepRL

SheepRL adds `ObservationNoiseWrapper` in `sheeprl/envs/wrappers.py`. The wrapper:

- Requires a Gymnasium `Dict` observation space.
- Stores `keys`, `noise_type`, `sigma`, `dropout_p`, `pink_alpha`, `pink_mix`, and `rng = np.random.default_rng(seed)`.
- Supports `gaussian`, `dropout`, and `pink`, but COMP9991 only needs Gaussian and pink.
- Copies the incoming observation dict with `obs = {k: np.array(v, copy=True) for k, v in observation.items()}`.
- Iterates only over configured `keys`; missing keys and non-`np.ndarray` values are skipped.
- Casts each selected key to `np.float32` before applying noise.
- If the selected key has a Gymnasium `Box` space, clips to `space.low` and `space.high`, then casts to `space.dtype`.
- Writes the result back to `obs[k]` and returns the modified observation dict.

SheepRL applies this wrapper in `sheeprl/utils/env.py` after image transform, frame stack, actions-as-observation, and reward-as-observation wrappers, and before seeding action/observation spaces, time limit, episode statistics, and video recording. If `obs_noise.obs_keys` is empty, SheepRL uses all configured encoder MLP and CNN keys; otherwise it uses `obs_noise.obs_keys`. The dumped DMC configs use `obs_keys: [rgb]`.

### Gaussian algorithm

For each selected observation key `k`:

```python
x = obs[k]
x = x.astype(np.float32)
noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
x = x + noise
x = np.clip(x, space.low, space.high)
x = x.astype(space.dtype)
obs[k] = x
```

Formula:

```text
eps_t[k] ~ Normal(0, sigma), independently for every pixel/channel element
y_t[k] = astype_space_dtype(clip(float32(x_t[k]) + float32(eps_t[k]), low_k, high_k))
```

Sigma semantics:

- `sigma` is in raw observation units, not normalized image units.
- For `uint8` RGB images with Box bounds `[0, 255]`, `sigma=20` means a standard deviation of 20 pixel values.
- SheepRL's dumped Gaussian DMC config sets `sigma: 5.0`; the COMP9991 `g20` condition must intentionally use `sigma: 20.0`.

Clipping and dtype conversion:

- Clipping happens after noise addition and before dtype conversion.
- For `uint8` image spaces, the final conversion is `x.astype(np.uint8)`.
- There is no explicit rounding step. NumPy integer `astype` truncates fractional values toward zero; after clipping to `[0, 255]`, this is equivalent to flooring positive fractional pixel values.

RNG use:

- A single `np.random.default_rng(seed)` is created when the wrapper is constructed.
- Gaussian draws consume that persistent RNG stream.
- The RNG is not reseeded on episode reset.

### Pink algorithm

SheepRL describes the pink mode as a "lightweight AR-style temporally correlated approximation." It is not a frequency-domain pink-noise filter.

State:

- `self._pink_state` is a dict keyed by observation key.
- For each selected key, the state shape exactly matches the selected observation array shape.
- State dtype is `np.float32`.
- If a key is unseen or its shape changes, state initializes to `np.zeros_like(x, dtype=np.float32)`.

Episode-reset behavior:

- On wrapper `reset()`, SheepRL clears the whole pink-state dict with `self._pink_state = {}`.
- It then calls the underlying env reset and immediately returns `self.observation(obs)`.
- Therefore, the reset observation is still noised; it uses a freshly zero-initialized pink state.
- The RNG is not reseeded by wrapper reset. The `seed` argument is passed through to the underlying env only.

For each selected key `k`, after `x = obs[k].astype(np.float32)`:

```python
if key not in pink_state or pink_state[key].shape != x.shape:
  pink_state[key] = np.zeros_like(x, dtype=np.float32)

prev = pink_state[key]
eps = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
state = pink_alpha * prev + (1.0 - pink_alpha) * eps
pink_state[key] = state
noise = pink_mix * state + (1.0 - pink_mix) * eps
x = x + noise
x = np.clip(x, space.low, space.high)
x = x.astype(space.dtype)
obs[k] = x
```

Formula:

```text
eps_t[k] ~ Normal(0, sigma), independently for every pixel/channel element
s_0[k] = zeros_like(x_0[k], float32) after reset or shape change
s_t[k] = pink_alpha * s_{t-1}[k] + (1 - pink_alpha) * eps_t[k]
n_t[k] = pink_mix * s_t[k] + (1 - pink_mix) * eps_t[k]
y_t[k] = astype_space_dtype(clip(float32(x_t[k]) + n_t[k], low_k, high_k))
```

Default/configured pink parameters in the dumped DMC pink config:

- `sigma: 5.0`
- `pink_alpha: 0.9`
- `pink_mix: 1.0`

With `pink_mix=1.0`, the applied noise is the AR state `s_t`. Since the recurrence scales innovations by `(1 - pink_alpha)`, `sigma` is the standard deviation of the white innovation `eps_t`, not the final stationary noise standard deviation. At reset with zero previous state, the first applied pink noise is `(1 - pink_alpha) * eps_t`.

RNG use:

- Pink uses the same persistent wrapper RNG as Gaussian.
- It draws one full `eps` tensor per selected key per observation call.
- Draw order follows the configured key order.
- Reset clears temporal state but does not reset the RNG stream.

## DreamerV3 Facts

Relevant facts from the audit and repository:

- `dreamerv3.main.make_env(config, index, **overrides)` constructs the suite env, then calls `wrap_env(env, config)`.
- `make_env()` only passes a `seed` constructor kwarg when the suite config contains `use_seed: True`.
- The pinned `dmc_vision` path uses `env.dmc` defaults without `use_seed: True`, so it does not currently pass a seed into `embodied.envs.dmc.DMC`.
- Current `wrap_env()` order is continuous-action `NormalizeAction`, `UnifyDtypes`, `CheckSpaces`, then continuous-action `ClipAction`.
- `UnifyDtypes` keeps `uint8` observations as `np.uint8`, converts floating observations to `np.float32`, and integer observations to `np.int32`.
- `CheckSpaces` validates observations against `obs_space`.
- DMC visual observations use key `image` when `env.dmc.image: True`, dtype `np.uint8`, and shape `(64, 64, 3)`.
- `dmc_vision` sets `task: dmc_walker_walk`, `env.dmc.proprio: False`, and keeps image observations enabled.
- Replay stores transitions from the driver before training.
- The agent encoder receives all observation keys except `is_first`, `is_last`, `is_terminal`, and `reward`.
- The RSSM encoder asserts image inputs are `jnp.uint8`, concatenates image keys, casts to compute dtype, divides by 255, and subtracts 0.5.
- Reconstruction targets are built from the same replay observations; image targets are cast to float and divided by 255.

Implication: the noise wrapper must corrupt the environment observation before replay insertion. Encoder-internal noise would not be parity-equivalent because replay and reconstruction targets would stay clean.

## Danijar Implementation Recommendations

### Placement

Add the future wrapper after `UnifyDtypes` and before `CheckSpaces`:

```python
env = embodied.wrappers.UnifyDtypes(env)
env = embodied.wrappers.ObservationNoise(env, ...)
env = embodied.wrappers.CheckSpaces(env)
```

This placement ensures corrupted `uint8` images enter:

- replay transitions,
- encoder input,
- decoder reconstruction targets,
- normal space validation.

It also preserves the existing action wrappers and avoids touching agent losses or model code.

### Observation-key behavior

Recommended Danijar semantics:

- Only apply noise to configured image keys.
- For DMC visual walker-walk, the default selected key should be `image`.
- A key is eligible only if it exists in `env.obs_space`, has dtype `np.uint8`, and has rank 3.
- Preserve all non-image observation keys exactly: same value, dtype, shape, and meaning.
- Preserve image shape and final dtype exactly.
- Keep `obs_space` unchanged.
- If noise is disabled, do not wrap the env at all, or use a strict identity wrapper whose outputs compare equal to the unwrapped env.

SheepRL copies every observation value before modifying selected keys. For DreamerV3, prefer a shallow dict copy and replacement of selected image keys only, so non-image values are not needlessly converted or copied.

### Algorithm parity mapping

In the Danijar `embodied.Env` interface, implement the noise in the wrapper's `step(action)` method:

1. Call `obs = self.env.step(action)`.
2. Use the returned `obs["is_first"]` as the authoritative episode-start event.
3. If pink noise is enabled and `obs["is_first"]` is true, clear the pink-state dict after `env.step()` and before applying noise to that first observation.
4. Apply the selected SheepRL formula to configured image keys only.
5. Return the observation dict with selected images replaced and all other keys preserved exactly.

Do not independently inspect the incoming action for pink-state clearing. That would duplicate reset logic already represented by `obs["is_first"]` and can introduce NumPy truth-value ambiguity. Do not reseed the wrapper RNG on reset.

Gaussian mapping:

```python
image = obs[key].astype(np.float32)
noise = self.rng.normal(0.0, sigma, image.shape).astype(np.float32)
image = image + noise
image = np.clip(image, space.low, space.high).astype(space.dtype)
```

Pink mapping:

```python
image = obs[key].astype(np.float32)
prev = self._pink_state.get(key)
if prev is None or prev.shape != image.shape:
  prev = np.zeros_like(image, dtype=np.float32)
eps = self.rng.normal(0.0, sigma, image.shape).astype(np.float32)
state = pink_alpha * prev + (1.0 - pink_alpha) * eps
self._pink_state[key] = state
noise = pink_mix * state + (1.0 - pink_mix) * eps
image = np.clip(image + noise, space.low, space.high).astype(space.dtype)
```

Do not normalize images before adding noise. Do not change the encoder path. Do not change replay storage. The only semantic change in noisy conditions is the raw `uint8` image value emitted by the environment wrapper.

### RNG derivation

Use deterministic per-environment wrapper-only RNG derivation in `make_env()`:

- The noise seed must not be described as derived from an existing DMC env seed.
- Do not alter DMC constructor kwargs or environment randomness in this feature commit.
- Treat environment seeding as a separate future audit, not part of the noise wrapper.
- Derive the wrapper seed deterministically from `config.seed`, environment `index`, and one fixed integer namespace.
- Keep one persistent `np.random.default_rng(noise_seed)` inside each wrapper instance.
- Do not reseed the wrapper RNG on episode reset.
- Derive distinct wrapper seeds for different environment indices.
- Recreating the same run with the same `config.seed`, env index, and namespace must reproduce the same noise stream.

Recommended derivation:

```python
OBS_NOISE_SEED_NAMESPACE = 0x0B50_5015
sequence = np.random.SeedSequence([
    int(config.seed),
    int(index),
    OBS_NOISE_SEED_NAMESPACE,
])
noise_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
```

Do not use process-randomized hashing for new noise seeds. This wrapper-only seed is independent of the pinned DMC constructor path.

### Configuration conditions

Add a small default `obs_noise` config namespace and three condition configs in `dreamerv3/configs.yaml`.

Recommended defaults:

```yaml
obs_noise:
  enabled: False
  type: gaussian
  keys: []
  sigma: 0.0
  pink_alpha: 0.9
  pink_mix: 1.0
```

Condition configs:

```yaml
clean:
  obs_noise.enabled: False

g20:
  obs_noise.enabled: True
  obs_noise.type: gaussian
  obs_noise.keys: [image]
  obs_noise.sigma: 20.0

pink:
  obs_noise.enabled: True
  obs_noise.type: pink
  obs_noise.keys: [image]
  obs_noise.sigma: 5.0
  obs_noise.pink_alpha: 0.9
  obs_noise.pink_mix: 1.0
```

The clean condition must be behaviorally identical to noise disabled. The `g20` and `pink` conditions should affect train and eval envs consistently unless a later experiment explicitly separates them.

### Exact files and functions for a later implementation

Modify only these repository files in the later implementation pass:

- `embodied/core/wrappers.py`
  - Add a new wrapper class, recommended name `ObservationNoise`.
  - Implement `__init__()`, `obs_space`, `step()`, `_apply_gaussian()`, `_apply_pink()`, and a small selected-key validator.
  - Match SheepRL Gaussian and pink formulas, clipping, dtype conversion, persistent RNG, and reset-state behavior.
- `dreamerv3/main.py`
  - In `make_env()`, derive a wrapper-only noise seed from `config.seed`, env index, and the fixed integer namespace.
  - Pass the env index or derived wrapper seed into `wrap_env()`.
  - Insert `ObservationNoise` immediately after `UnifyDtypes` and before `CheckSpaces` when `config.obs_noise.enabled` is true.
  - Do not change DMC constructor kwargs and do not add `use_seed` handling as part of this feature.
- `dreamerv3/configs.yaml`
  - Add the `obs_noise` default namespace.
  - Add condition configs `clean`, `g20`, and `pink`.
- `embodied/tests/test_observation_noise.py`
  - Add focused wrapper tests listed below.

Do not modify `dreamerv3/agent.py`, `dreamerv3/rssm.py`, optimizer code, replay code, or dependencies for this feature.

## Unit Test Plan

Add wrapper-level tests with a tiny deterministic repository-native `embodied.Env`. The dummy env must expose `obs_space` and `act_space` using `elements.Space`, not Gymnasium spaces. It should emit:

- `image`: `elements.Space(np.uint8, (H, W, 3), 0, 255)`
- at least one non-image key, such as a float vector
- standard Dreamer control keys if needed by the dummy env

Required tests:

1. Identity when disabled
   - With `obs_noise.enabled: False`, `wrap_env()` should not alter observations.
   - Clean disabled mode must not instantiate the noise wrapper.
   - Clean images, non-image keys, dtypes, and shapes must match the baseline env.

2. Shape and `uint8` preservation
   - With Gaussian and pink enabled on `image`, every emitted image remains the same shape and `np.uint8`.
   - `obs_space['image']` remains unchanged.
   - Non-image observation keys remain exactly unchanged.

3. Clipping
   - Use a controlled RNG or monkeypatch the wrapper RNG to return known negative and positive noise values.
   - Verify values below 0 clip to 0 and values above 255 clip to 255 before conversion to `uint8`.
   - Verify fractional positive values are truncated by `astype(np.uint8)`, not rounded.

4. Same-seed reproducibility
   - Two wrapper instances with the same `config.seed`, env index, fixed namespace, config, and clean observation sequence must emit identical corrupted images.
   - Cover Gaussian and pink.

5. Different-seed divergence
   - Two wrapper instances with different derived seeds must emit different corrupted images for the same clean observation sequence.
   - Use a large enough image or multiple steps to make accidental equality negligible.

6. Pink temporal correlation
   - For a constant clean image stream, pink noise should show positive lag-1 temporal correlation.
   - Prefer checking the wrapper's float pink state or a controlled scalar sequence to avoid fragile assertions after `uint8` clipping/truncation.
   - Confirm the recurrence is `state = alpha * prev + (1 - alpha) * eps`.

7. Pink reset behavior
   - After reset, `_pink_state` is cleared.
   - The reset observation is still noised using zero previous state.
   - The wrapper RNG is not reseeded by reset.
   - With controlled `eps`, first pink noise after reset equals `(1 - alpha) * eps` when `pink_mix=1.0`.

8. Multi-environment seed separation
   - Env index 0 and env index 1 with the same `config.seed` must derive different wrapper seeds and produce different noise streams.
   - Recreating env index 0 with the same `config.seed` must reproduce index 0's stream exactly.

9. Golden SheepRL parity
   - Use fixed images, fixed wrapper seeds, and fixed controlled noise draws for the first `N` observations.
   - Compute expected Gaussian outputs with the exact SheepRL NumPy formula: cast image to `np.float32`, add `rng.normal(0.0, sigma, image.shape).astype(np.float32)`, clip to the `elements.Space` low/high bounds, and cast to `np.uint8`.
   - Compute expected pink outputs with the exact SheepRL NumPy formula: zero state after first-observation reset, draw `eps`, update `state = pink_alpha * prev + (1.0 - pink_alpha) * eps`, compute `noise = pink_mix * state + (1.0 - pink_mix) * eps`, add, clip, and cast to `np.uint8`.
   - Require elementwise equality between wrapper outputs and expected arrays after clipping and `uint8` conversion for the first `N` observations.
   - Verify clean disabled mode does not instantiate the wrapper.

## Engineering Smoke Plan

Run short engineering-only DMC visual walker-walk smokes after implementation. These are not scientific comparisons. Each smoke must use a unique logdir under project scratch.

Recommended 10k-step smoke set on one GPU:

```sh
python dreamerv3/main.py --configs dmc_vision size1m clean --task dmc_walker_walk --seed 0 --run.steps 10000 --run.envs 4 --run.train_ratio 32 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-smoke/clean
python dreamerv3/main.py --configs dmc_vision size1m g20 --task dmc_walker_walk --seed 0 --run.steps 10000 --run.envs 4 --run.train_ratio 32 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-smoke/g20
python dreamerv3/main.py --configs dmc_vision size1m pink --task dmc_walker_walk --seed 0 --run.steps 10000 --run.envs 4 --run.train_ratio 32 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-smoke/pink
```

Fallback 5k-step smoke if compile/runtime cost is high:

```sh
python dreamerv3/main.py --configs dmc_vision size1m clean --task dmc_walker_walk --seed 0 --run.steps 5000 --run.envs 2 --run.train_ratio 16 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-smoke/clean-5k
python dreamerv3/main.py --configs dmc_vision size1m g20 --task dmc_walker_walk --seed 0 --run.steps 5000 --run.envs 2 --run.train_ratio 16 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-smoke/g20-5k
python dreamerv3/main.py --configs dmc_vision size1m pink --task dmc_walker_walk --seed 0 --run.steps 5000 --run.envs 2 --run.train_ratio 16 --run.save_every 0 --run.log_every 5 --run.report_every 30 --jax.prealloc False --logdir ./scratch/comp9991-smoke/pink-5k
```

Smoke checks:

- All three conditions start, compile, and train without space-check failures.
- Replay receives `uint8` images under all three conditions.
- Existing train metrics remain present.
- No new loss terms, gates, thresholds, or normalizer updates are introduced.
- Optional manual inspection of logged videos or sampled replay images should show clean, Gaussian, and temporally correlated pink corruption patterns.

## Hard Prohibitions

The later implementation must explicitly avoid:

- Disagreement instrumentation.
- Teacher gating.
- Actor loss changes.
- Value loss changes.
- World-model loss changes.
- Return normalization changes.
- Threshold tuning.
- Any dependency changes.
- Any training behavior changes beyond the selected observation image corruption.
