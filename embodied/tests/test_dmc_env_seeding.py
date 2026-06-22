import json
import pathlib
import types

import elements
import embodied
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3 import main as dreamer_main


class _DummyEnv(embodied.Env):

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (2, 2, 3), 0, 255),
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
    return {
        'image': np.zeros((2, 2, 3), np.uint8),
        'reward': np.float32(0),
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


def _config(tmpdir, **kwargs):
  values = dict(
      task='dmc_walker_walk',
      seed=5,
      logdir=str(tmpdir),
      script='train',
      run=dict(envs=3),
      env=dict(dmc=dict(size=(64, 64), repeat=1, proprio=False,
                        image=True, camera=-1)),
      obs_noise=dict(enabled=False, type='gaussian', keys=('image',),
                     sigma=0.0, pink_alpha=0.9, pink_mix=1.0),
  )
  values.update(kwargs)
  return elements.Config(values)


def _has_noise_wrapper(env):
  while isinstance(env, embodied.wrappers.Wrapper):
    if isinstance(env, embodied.wrappers.ObservationNoise):
      return True
    env = env.env
  return False


def _patch_dmc_wrapping(monkeypatch, dmc_module):
  monkeypatch.setattr(dmc_module.from_dm, 'FromDM', lambda env: env)
  monkeypatch.setattr(
      dmc_module.embodied.wrappers, 'ActionRepeat',
      lambda env, repeat: env)


class TestDMCEnvSeeding:

  def test_seed_derivation_reproducible_and_separated(self):
    assert dreamer_main._dmc_env_seed(7, 3) == dreamer_main._dmc_env_seed(7, 3)
    assert dreamer_main._dmc_env_seed(7, 3) != dreamer_main._dmc_env_seed(8, 3)
    assert dreamer_main._dmc_env_seed(7, 3) != dreamer_main._dmc_env_seed(7, 4)
    assert dreamer_main.DMC_ENV_SEED_NAMESPACE != (
        dreamer_main.OBS_NOISE_SEED_NAMESPACE)
    assert dreamer_main._dmc_env_seed(7, 3) != (
        dreamer_main._obs_noise_seed(7, 3))

  def test_noise_conditions_do_not_alter_dmc_seed(self):
    path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(path.read_text())
    base = elements.Config(configs['defaults'])
    expected = dreamer_main._dmc_env_seed(base.seed, 0)
    for condition in ('clean', 'g20', 'pink'):
      config = base.update(configs[condition])
      assert dreamer_main._dmc_env_seed(config.seed, 0) == expected

  def test_make_env_passes_dmc_seed(self, monkeypatch, tmp_path):
    calls = []

    def fake_import_module(name):
      assert name == 'embodied.envs.dmc'

      class FakeDMC:

        def __init__(self, task, **kwargs):
          calls.append((task, kwargs))

      return types.SimpleNamespace(DMC=FakeDMC)

    monkeypatch.setattr(dreamer_main.importlib, 'import_module',
                        fake_import_module)
    monkeypatch.setattr(dreamer_main, 'wrap_env',
                        lambda env, config, index=0: env)
    config = _config(tmp_path, seed=11)

    dreamer_main.make_env(config, 2)

    assert calls == [(
        'walker_walk',
        {
            'size': (64, 64),
            'repeat': 1,
            'proprio': False,
            'image': True,
            'camera': -1,
            'seed': dreamer_main._dmc_env_seed(11, 2),
        },
    )]

  def test_make_env_preserves_non_dmc_use_seed(self, monkeypatch, tmp_path):
    calls = []

    def fake_import_module(name):
      assert name == 'embodied.envs.dummy'

      class FakeDummy:

        def __init__(self, task, **kwargs):
          calls.append((task, kwargs))

      return types.SimpleNamespace(Dummy=FakeDummy)

    monkeypatch.setattr(dreamer_main.importlib, 'import_module',
                        fake_import_module)
    monkeypatch.setattr(dreamer_main, 'wrap_env',
                        lambda env, config, index=0: env)
    config = _config(
        tmp_path,
        task='dummy_disc',
        env=dict(dummy=dict(use_seed=True)),
        seed=11)

    dreamer_main.make_env(config, 2)

    assert calls[0][0] == 'disc'
    assert 'seed' in calls[0][1]
    assert calls[0][1]['seed'] == hash((11, 2)) % (2 ** 32 - 1)

  def test_suite_load_seeded_and_unseeded_calls(self, monkeypatch):
    from embodied.envs import dmc

    _patch_dmc_wrapping(monkeypatch, dmc)
    calls = []
    monkeypatch.setattr(
        dmc.suite, 'load',
        lambda *args, **kwargs: calls.append((args, kwargs)) or object())

    dmc.DMC('walker_walk', seed=123)
    dmc.DMC('walker_walk', seed=None)

    assert calls[0] == (
        ('walker', 'walk'),
        {'task_kwargs': {'random': 123}},
    )
    assert calls[1] == (('walker', 'walk'), {})

  def test_manipulation_and_rodent_paths_unchanged(self, monkeypatch):
    from embodied.envs import dmc

    _patch_dmc_wrapping(monkeypatch, dmc)
    suite_calls = []
    manip_calls = []
    rodent_calls = []

    monkeypatch.setattr(
        dmc.suite, 'load',
        lambda *args, **kwargs: suite_calls.append((args, kwargs)) or object())
    monkeypatch.setattr(
        dmc.manipulation, 'load',
        lambda name: manip_calls.append(name) or object())
    monkeypatch.setattr(
        dmc.basic_rodent_2020, 'escape',
        lambda: rodent_calls.append('escape') or object(),
        raising=False)

    dmc.DMC('manip_reach', seed=123)
    dmc.DMC('rodent_escape', seed=123)

    assert suite_calls == []
    assert manip_calls == ['reach_vision']
    assert rodent_calls == ['escape']

  def test_metadata_contains_recomputed_mapping(self, tmp_path):
    config = _config(tmp_path, seed=13, run=dict(envs=4))

    dreamer_main._write_dmc_env_seed_metadata(config, tmp_path)

    metadata = json.loads((tmp_path / 'dmc_env_seeds.json').read_text())
    assert metadata == {
        'derivation': dreamer_main.DMC_ENV_SEED_DERIVATION,
        'derivation_version': dreamer_main.DMC_ENV_SEED_DERIVATION_VERSION,
        'envs': {
            str(index): dreamer_main._dmc_env_seed(13, index)
            for index in range(4)
        },
        'namespace': dreamer_main.DMC_ENV_SEED_NAMESPACE,
        'script': 'train',
        'seed': 13,
    }

  def test_metadata_not_written_by_make_env(self, monkeypatch, tmp_path):
    monkeypatch.setattr(
        dreamer_main.importlib, 'import_module',
        lambda name: types.SimpleNamespace(
            DMC=lambda task, **kwargs: _DummyEnv()))
    monkeypatch.setattr(dreamer_main, 'wrap_env',
                        lambda env, config, index=0: env)
    config = _config(tmp_path, seed=13)

    dreamer_main.make_env(config, 0)

    assert not (tmp_path / 'dmc_env_seeds.json').exists()

  def test_clean_mode_does_not_instantiate_observation_noise(self, tmp_path):
    config = _config(tmp_path)

    env = dreamer_main.wrap_env(_DummyEnv(), config, index=0)

    assert not _has_noise_wrapper(env)
