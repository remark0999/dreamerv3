import pathlib

import elements
import embodied
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3 import main as dreamer_main


class _ImageEnv(embodied.Env):

  def __init__(self, images=None, firsts=None):
    self.images = list(images or [np.full((2, 3, 1), 100, np.uint8)])
    self.firsts = list(firsts or [True] + [False] * (len(self.images) - 1))
    self.index = 0
    self.last_vector = None

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self.images[0].shape, 0, 255),
        'vector': elements.Space(np.float32, (2,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {'reset': elements.Space(bool)}

  def step(self, action):
    del action
    index = min(self.index, len(self.images) - 1)
    self.index += 1
    self.last_vector = np.array([index, index + 1], np.float32)
    return {
        'image': np.array(self.images[index], copy=True),
        'vector': self.last_vector,
        'reward': np.float32(index),
        'is_first': bool(self.firsts[index]),
        'is_last': False,
        'is_terminal': False,
    }


class _FixedRng:

  def __init__(self, draws, expected_loc=None, expected_scale=None):
    self.draws = [np.asarray(x) for x in draws]
    self.expected_loc = expected_loc
    self.expected_scale = expected_scale
    self.index = 0

  def normal(self, loc=0.0, scale=1.0, size=None):
    if self.expected_loc is not None:
      assert loc == self.expected_loc
    if self.expected_scale is not None:
      assert scale == self.expected_scale
    draw = self.draws[self.index]
    self.index += 1
    assert tuple(draw.shape) == tuple(size), (draw.shape, size)
    return draw


def _action():
  return {'reset': False}


def _noise(env, noise_type='gaussian', seed=0, sigma=1.0, **kwargs):
  return embodied.wrappers.ObservationNoise(
      env, keys=['image'], noise_type=noise_type, sigma=sigma, seed=seed,
      **kwargs)


def _has_noise_wrapper(env):
  while isinstance(env, embodied.wrappers.Wrapper):
    if isinstance(env, embodied.wrappers.ObservationNoise):
      return True
    env = env.env
  return False


def _config(enabled, **kwargs):
  obs_noise = dict(
      enabled=enabled, type='gaussian', keys=['image'], sigma=0.0,
      pink_alpha=0.9, pink_mix=1.0)
  obs_noise.update(kwargs)
  return elements.Config(seed=5, obs_noise=obs_noise)


def _expected_gaussian(images, draws, space):
  outputs = []
  for image, draw in zip(images, draws):
    image = image.astype(np.float32)
    noise = draw.astype(np.float32)
    image = image + noise
    outputs.append(np.clip(image, space.low, space.high).astype(space.dtype))
  return outputs


def _expected_pink(images, draws, firsts, space, alpha=0.9, mix=1.0):
  outputs = []
  state = None
  for image, draw, is_first in zip(images, draws, firsts):
    image = image.astype(np.float32)
    if is_first:
      state = None
    if state is None or state.shape != image.shape:
      state = np.zeros_like(image, dtype=np.float32)
    eps = draw.astype(np.float32)
    state = alpha * state + (1.0 - alpha) * eps
    noise = mix * state + (1.0 - mix) * eps
    outputs.append(
        np.clip(image + noise, space.low, space.high).astype(space.dtype))
  return outputs


class TestObservationNoise:

  def test_disabled_identity_and_wrapper_absence(self):
    images = [np.arange(6, dtype=np.uint8).reshape((2, 3, 1))]
    baseline = _ImageEnv(images)
    wrapped = dreamer_main.wrap_env(
        _ImageEnv(images), _config(False), index=0)
    assert not _has_noise_wrapper(wrapped)

    expected = baseline.step(_action())
    actual = wrapped.step(_action())
    assert actual.keys() == expected.keys()
    for key in expected:
      np.testing.assert_array_equal(actual[key], expected[key])

  def test_shape_dtype_and_nonselected_preservation(self):
    image = np.full((2, 3, 1), 100, np.uint8)
    for noise_type in ('gaussian', 'pink'):
      env = _ImageEnv([image])
      wrapped = _noise(env, noise_type=noise_type, sigma=2.0)
      obs_space = wrapped.obs_space
      assert obs_space.keys() == env.obs_space.keys()
      assert obs_space['image'].shape == env.obs_space['image'].shape
      assert obs_space['image'].dtype == env.obs_space['image'].dtype
      obs = wrapped.step(_action())
      assert obs['image'].shape == image.shape
      assert obs['image'].dtype == np.uint8
      assert obs['vector'] is env.last_vector
      np.testing.assert_array_equal(obs['vector'], np.array([0, 1], np.float32))

  def test_clipping_and_truncation(self):
    image = np.array([[[1], [250], [1]]], np.uint8)
    draw = np.array([[[-2.2], [10.9], [1.9]]], np.float32)
    env = _ImageEnv([image])
    wrapped = _noise(env, sigma=1.0)
    wrapped._rng = _FixedRng([draw])

    obs = wrapped.step(_action())
    expected = np.array([[[0], [255], [2]]], np.uint8)
    np.testing.assert_array_equal(obs['image'], expected)

  def test_same_seed_reproducibility(self):
    images = [np.full((4, 4, 3), 127, np.uint8) for _ in range(4)]
    for noise_type in ('gaussian', 'pink'):
      left = _noise(_ImageEnv(images, [True, False, False, False]),
                    noise_type=noise_type, seed=12, sigma=5.0)
      right = _noise(_ImageEnv(images, [True, False, False, False]),
                     noise_type=noise_type, seed=12, sigma=5.0)
      for _ in images:
        np.testing.assert_array_equal(
            left.step(_action())['image'],
            right.step(_action())['image'])

  def test_different_seed_divergence(self):
    image = np.full((16, 16, 3), 127, np.uint8)
    for noise_type in ('gaussian', 'pink'):
      left = _noise(
          _ImageEnv([image]), noise_type=noise_type, seed=1, sigma=20.0)
      right = _noise(
          _ImageEnv([image]), noise_type=noise_type, seed=2, sigma=20.0)
      assert not np.array_equal(
          left.step(_action())['image'],
          right.step(_action())['image'])

  def test_pink_temporal_correlation(self):
    images = [np.full((1, 1, 1), 100, np.uint8) for _ in range(6)]
    draws = [np.full((1, 1, 1), 10, np.float32)] * len(images)
    wrapped = _noise(
        _ImageEnv(images, [True] + [False] * (len(images) - 1)),
        noise_type='pink', sigma=1.0, pink_alpha=0.9, pink_mix=1.0)
    wrapped._rng = _FixedRng(draws)

    states = []
    for _ in images:
      wrapped.step(_action())
      states.append(float(wrapped._pink_state['image'][0, 0, 0]))
    corr = np.corrcoef(states[:-1], states[1:])[0, 1]
    assert corr > 0
    np.testing.assert_allclose(states[:3], [1.0, 1.9, 2.71], rtol=1e-6)

  def test_pink_state_reset_without_rng_reset(self):
    images = [np.full((1, 1, 1), 100, np.uint8) for _ in range(3)]
    firsts = [True, False, True]
    draws = [
        np.full((1, 1, 1), 10, np.float32),
        np.full((1, 1, 1), 20, np.float32),
        np.full((1, 1, 1), 30, np.float32),
    ]
    wrapped = _noise(
        _ImageEnv(images, firsts), noise_type='pink', sigma=1.0,
        pink_alpha=0.9, pink_mix=1.0)
    wrapped._rng = _FixedRng(draws)

    wrapped.step(_action())
    assert wrapped._rng.index == 1
    np.testing.assert_allclose(wrapped._pink_state['image'], [[[1.0]]])
    wrapped.step(_action())
    assert wrapped._rng.index == 2
    np.testing.assert_allclose(wrapped._pink_state['image'], [[[2.9]]])
    wrapped.step(_action())
    assert wrapped._rng.index == 3
    np.testing.assert_allclose(wrapped._pink_state['image'], [[[3.0]]])

  def test_per_environment_seed_separation(self):
    seed0 = dreamer_main._obs_noise_seed(5, 0)
    seed1 = dreamer_main._obs_noise_seed(5, 1)
    assert seed0 != seed1
    assert seed0 == dreamer_main._obs_noise_seed(5, 0)

    image = np.full((8, 8, 3), 127, np.uint8)
    env0a = _noise(_ImageEnv([image]), seed=seed0, sigma=10.0)
    env0b = _noise(_ImageEnv([image]), seed=seed0, sigma=10.0)
    env1 = _noise(_ImageEnv([image]), seed=seed1, sigma=10.0)
    out0a = env0a.step(_action())['image']
    out0b = env0b.step(_action())['image']
    out1 = env1.step(_action())['image']
    np.testing.assert_array_equal(out0a, out0b)
    assert not np.array_equal(out0a, out1)

  def test_exact_gaussian_golden_parity(self):
    images = [
        np.array([[[10], [250]]], np.uint8),
        np.array([[[0], [128]]], np.uint8),
    ]
    draws = [
        np.array([[[1.7], [10.1]]], np.float32),
        np.array([[[-5.0], [0.9]]], np.float32),
    ]
    env = _ImageEnv(images, [True, False])
    wrapped = _noise(env, sigma=20.0, seed=7)
    wrapped._rng = _FixedRng(draws, expected_loc=0.0, expected_scale=20.0)
    expected = _expected_gaussian(images, draws, env.obs_space['image'])

    actual = [wrapped.step(_action())['image'] for _ in images]
    for lhs, rhs in zip(actual, expected):
      np.testing.assert_array_equal(lhs, rhs)

  def test_exact_pink_golden_parity(self):
    images = [
        np.array([[[10], [250]]], np.uint8),
        np.array([[[0], [128]]], np.uint8),
        np.array([[[5], [128]]], np.uint8),
    ]
    firsts = [True, False, True]
    draws = [
        np.array([[[10], [10]]], np.float32),
        np.array([[[20], [20]]], np.float32),
        np.array([[[30], [30]]], np.float32),
    ]
    env = _ImageEnv(images, firsts)
    wrapped = _noise(
        env, noise_type='pink', sigma=5.0, seed=7,
        pink_alpha=0.9, pink_mix=1.0)
    wrapped._rng = _FixedRng(draws, expected_loc=0.0, expected_scale=5.0)
    expected = _expected_pink(
        images, draws, firsts, env.obs_space['image'], alpha=0.9, mix=1.0)

    actual = [wrapped.step(_action())['image'] for _ in images]
    for lhs, rhs in zip(actual, expected):
      np.testing.assert_array_equal(lhs, rhs)


  def test_dropout_zeroes_image_and_logs(self):
    image = np.full((2, 3, 1), 100, np.uint8)
    env = _ImageEnv([image])
    wrapped = embodied.wrappers.ObservationNoise(
        env, ['image'], noise_type='dropout', dropout_value=0)

    obs = wrapped.step(_action())

    np.testing.assert_array_equal(obs['image'], np.zeros_like(image))
    assert int(obs['log/obs_noise_code']) == 2
    assert float(obs['log/obs_noise_is_corrupt']) == 1.0
    assert float(obs['log/obs_noise_is_dropout']) == 1.0
    assert float(obs['log/obs_noise_is_pink']) == 0.0
    assert float(obs['log/obs_noise_input_absdiff']) == 100.0

  def test_mixdrop_forced_dropout_and_clean_modes(self):
    image = np.full((2, 3, 1), 100, np.uint8)

    drop = embodied.wrappers.ObservationNoise(
        _ImageEnv([image]), ['image'], noise_type='mixdrop',
        clean_prob=0.0, dropout_prob=1.0, dropout_value=0)
    obs = drop.step(_action())
    np.testing.assert_array_equal(obs['image'], np.zeros_like(image))
    assert int(obs['log/obs_noise_code']) == 2

    clean = embodied.wrappers.ObservationNoise(
        _ImageEnv([image]), ['image'], noise_type='mixdrop',
        clean_prob=1.0, dropout_prob=0.0, dropout_value=0)
    obs = clean.step(_action())
    np.testing.assert_array_equal(obs['image'], image)
    assert int(obs['log/obs_noise_code']) == 0
    assert float(obs['log/obs_noise_input_absdiff']) == 0.0

  def test_mixpinkdrop_forced_pink_logs(self):
    image = np.full((1, 1, 1), 100, np.uint8)
    wrapped = embodied.wrappers.ObservationNoise(
        _ImageEnv([image], [True]), ['image'], noise_type='mixpinkdrop',
        clean_prob=0.0, pink_prob=1.0, dropout_prob=0.0,
        sigma=1.0, pink_alpha=0.9, pink_mix=1.0)

    obs = wrapped.step(_action())

    assert int(obs['log/obs_noise_code']) == 1
    assert float(obs['log/obs_noise_is_pink']) == 1.0
    assert obs['image'].shape == image.shape
    assert obs['image'].dtype == np.uint8


  def test_validation_errors(self):
    env = _ImageEnv()
    with pytest.raises(ValueError, match='type'):
      embodied.wrappers.ObservationNoise(env, ['image'], noise_type='not_a_noise')
    with pytest.raises(ValueError, match='at least one'):
      embodied.wrappers.ObservationNoise(
          _ImageEnv(), [], noise_type='gaussian')
    with pytest.raises(KeyError, match='missing'):
      embodied.wrappers.ObservationNoise(env, ['missing'])
    with pytest.raises(ValueError, match='rank-3 uint8'):
      embodied.wrappers.ObservationNoise(env, ['vector'])
    with pytest.raises(ValueError, match='pink_alpha'):
      embodied.wrappers.ObservationNoise(env, ['image'], pink_alpha=1.1)
    with pytest.raises(ValueError, match='pink_mix'):
      embodied.wrappers.ObservationNoise(env, ['image'], pink_mix=-0.1)

  @pytest.mark.parametrize('sigma', [np.nan, np.inf, -np.inf, -1.0])
  def test_invalid_sigma(self, sigma):
    with pytest.raises(ValueError, match='finite and nonnegative'):
      embodied.wrappers.ObservationNoise(_ImageEnv(), ['image'], sigma=sigma)

  def test_elements_config_conditions_parse(self):
    path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(path.read_text())
    defaults = elements.Config(configs['defaults'])

    clean = defaults.update(configs['clean'])
    g20 = defaults.update(configs['g20'])
    pink = defaults.update(configs['pink'])

    assert clean.obs_noise.enabled is False
    assert g20.obs_noise.get('keys') == ('image',)
    assert g20.obs_noise.type == 'gaussian'
    assert g20.obs_noise.sigma == 20.0
    assert pink.obs_noise.get('keys') == ('image',)
    assert pink.obs_noise.type == 'pink'
    assert pink.obs_noise.sigma == 5.0

    mixdrop30 = defaults.update(configs['mixdrop30'])
    mixpinkdrop = defaults.update(configs['mixpinkdrop'])

    assert mixdrop30.obs_noise.enabled is True
    assert mixdrop30.obs_noise.type == 'mixdrop'
    assert mixdrop30.obs_noise.clean_prob == 0.70
    assert mixdrop30.obs_noise.dropout_prob == 0.30

    assert mixpinkdrop.obs_noise.enabled is True
    assert mixpinkdrop.obs_noise.type == 'mixpinkdrop'
    assert mixpinkdrop.obs_noise.clean_prob == 0.60
    assert mixpinkdrop.obs_noise.pink_prob == 0.25
    assert mixpinkdrop.obs_noise.dropout_prob == 0.15
    assert mixpinkdrop.obs_noise.sigma == 5.0
