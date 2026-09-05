import pathlib

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3 import agent as dreamer_agent


def _latent_config(**overrides):
  config = dict(
      enabled=True,
      target='deter',
      timing='start',
      prob=1.0,
      sigma=0.5,
      scale='rms',
      eps=1e-6,
      stop_grad_noise=True,
      apply_train_policy=True,
      apply_train_value=False,
      apply_report_shadow=True,
      apply_policy_eval=False,
      apply_env_acting=False)
  config.update(overrides)
  return elements.Config(latent_noise=config)


def _agent(**overrides):
  agent = object.__new__(dreamer_agent.Agent)
  agent.config = _latent_config(**overrides)
  return agent


def _starts():
  return {
      'deter': jnp.arange(12, dtype=jnp.float32).reshape((3, 4)) + 1.0,
      'stoch': jnp.arange(24, dtype=jnp.float32).reshape((3, 2, 4)),
      'logit': jnp.ones((3, 2), jnp.float32),
      'logits': jnp.ones((3, 2, 2), jnp.float32) * 2.0,
      'other': jnp.ones((3, 1), jnp.float32) * 3.0,
  }


def _apply(agent, starts):

  def call(starts):
    return dreamer_agent.Agent._apply_latent_noise(agent, starts)

  _, out = nj.pure(call)(
      {}, starts, seed=jnp.array([123, 456], jnp.uint32))
  return out


def _assert_tree_equal(left, right):
  jax.tree.map(
      lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)),
      left, right)


@pytest.mark.parametrize('overrides', [
    dict(enabled=False),
    dict(prob=0.0),
    dict(sigma=0.0),
])
def test_disabled_latent_noise_identity_cases(overrides):
  starts = _starts()
  out, metrics = _apply(_agent(**overrides), starts)

  _assert_tree_equal(out, starts)
  assert float(metrics['applied_frac']) == 0.0
  assert float(metrics['deter_absdiff_mean']) == 0.0
  assert float(metrics['deter_rel_rms_mean']) == 0.0


def test_prob_one_sigma_positive_changes_only_deter():
  starts = _starts()
  out, metrics = _apply(_agent(prob=1.0, sigma=0.25), starts)

  assert out['deter'].shape == starts['deter'].shape
  assert not np.array_equal(np.asarray(out['deter']), np.asarray(starts['deter']))
  for key in ('stoch', 'logit', 'logits', 'other'):
    np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(starts[key]))
    assert out[key].shape == starts[key].shape
  assert float(metrics['enabled']) == 1.0
  assert float(metrics['applied_frac']) == 1.0
  assert float(metrics['deter_absdiff_mean']) > 0.0
  assert float(metrics['deter_rel_rms_mean']) > 0.0


def test_shapes_are_preserved_and_input_is_not_mutated():
  starts = _starts()
  original = jax.tree.map(lambda x: jnp.array(x), starts)

  out, _ = _apply(_agent(prob=1.0, sigma=0.25), starts)

  assert {key: value.shape for key, value in out.items()} == {
      key: value.shape for key, value in starts.items()}
  _assert_tree_equal(starts, original)


@pytest.mark.parametrize('prob', [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_prob_rejected(prob):
  with pytest.raises(ValueError, match='latent_noise.prob'):
    dreamer_agent._validate_latent_noise_config(_latent_config(prob=prob))


@pytest.mark.parametrize('sigma', [-0.1, np.nan, np.inf, -np.inf])
def test_invalid_sigma_rejected(sigma):
  with pytest.raises(ValueError, match='latent_noise.sigma'):
    dreamer_agent._validate_latent_noise_config(_latent_config(sigma=sigma))


def test_invalid_target_rejected():
  with pytest.raises(ValueError, match='latent_noise.target'):
    dreamer_agent._validate_latent_noise_config(_latent_config(target='stoch'))


def test_invalid_scale_rejected():
  with pytest.raises(ValueError, match='latent_noise.scale'):
    dreamer_agent._validate_latent_noise_config(_latent_config(scale='std'))


@pytest.mark.parametrize('field', [
    'apply_train_value',
    'apply_env_acting',
    'apply_policy_eval',
])
def test_unsupported_application_flags_rejected(field):
  with pytest.raises(ValueError, match=field):
    dreamer_agent._validate_latent_noise_config(_latent_config(**{field: True}))


def test_lnd0_configs_exist_and_tg0_filters_keep_latent_noise():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  defaults = elements.Config(configs['defaults'])
  conditions = {
      'lnd0_clean': (False, 0.0),
      'lnd0_deter_p30_s005': (True, 0.05),
      'lnd0_deter_p30_s015': (True, 0.15),
      'lnd0_deter_p30_s030': (True, 0.30),
  }
  arms = ('tg0_baseline', 'tg0_signal', 'tg0_shuffle', 'tg0_uniform')

  for condition, (enabled, sigma) in conditions.items():
    cfg = defaults.update(configs[condition])
    assert cfg.obs_noise.enabled is False
    assert cfg.agent.latent_noise.enabled is enabled
    assert cfg.agent.latent_noise.target == 'deter'
    if enabled:
      assert cfg.agent.latent_noise.prob == 0.30
      assert cfg.agent.latent_noise.sigma == sigma
      assert cfg.agent.latent_noise.scale == 'rms'
      assert cfg.agent.latent_noise.apply_train_policy is True
      assert cfg.agent.latent_noise.apply_train_value is False
      assert cfg.agent.latent_noise.apply_report_shadow is True
      assert cfg.agent.latent_noise.apply_policy_eval is False
      assert cfg.agent.latent_noise.apply_env_acting is False
    assert 'latent_noise' in cfg.logger.filter

  for arm in arms:
    cfg = defaults.update(configs[arm])
    assert 'teacher_gate' in cfg.logger.filter
    assert 'latent_noise' in cfg.logger.filter

  for condition in conditions:
    for arm in arms:
      cfg = defaults
      for name in ('dmc_vision', 'size12m', 'shadow_disagreement',
                   condition, arm):
        cfg = cfg.update(configs[name])
      assert 'teacher_gate' in cfg.logger.filter
      assert 'latent_noise' in cfg.logger.filter
