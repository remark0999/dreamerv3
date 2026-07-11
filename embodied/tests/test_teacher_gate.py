import inspect
import pathlib

import elements
import jax.numpy as jnp
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3 import agent as dreamer_agent


def test_teacher_gate_alpha_bounds_and_direction():
  conflict = jnp.array([0.0, 0.1, 1.0, 2.0], jnp.float32)
  alpha = dreamer_agent._teacher_gate_alpha_from_conflict(
      conflict, beta=1.0, alpha_min=0.25, eps=1e-6)

  assert alpha.shape == conflict.shape
  assert float(alpha.min()) >= 0.25
  assert float(alpha.max()) <= 1.0
  assert float(alpha[-1]) < float(alpha[0])


def test_teacher_gate_constant_conflict_is_identity():
  conflict = jnp.ones((8,), jnp.float32) * 0.5
  alpha = dreamer_agent._teacher_gate_alpha_from_conflict(
      conflict, beta=0.5, alpha_min=0.25, eps=1e-6)

  np.testing.assert_allclose(np.asarray(alpha), np.ones((8,), np.float32))


def test_apply_teacher_gate_only_scales_policy_loss():
  policy_loss = jnp.ones((2, 3), jnp.float32)
  gate = jnp.array([[1.0], [0.5]], jnp.float32)

  gated = dreamer_agent._apply_teacher_gate_to_policy_loss(policy_loss, gate)

  expected = jnp.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]], jnp.float32)
  np.testing.assert_allclose(np.asarray(gated), np.asarray(expected))


def test_teacher_gate_disabled_returns_identity_policy_loss():
  policy_loss = jnp.arange(6, dtype=jnp.float32).reshape((2, 3))

  out = dreamer_agent._apply_teacher_gate_to_policy_loss(policy_loss, None)

  np.testing.assert_allclose(np.asarray(out), np.asarray(policy_loss))


def test_teacher_gate_config_validation_errors():
  with pytest.raises(ValueError, match='teacher_gate.mode'):
    dreamer_agent._validate_teacher_gate_config(
        elements.Config(teacher_gate=dict(enabled=True, mode='bad')))
  with pytest.raises(ValueError, match='teacher_gate.beta'):
    dreamer_agent._validate_teacher_gate_config(
        elements.Config(teacher_gate=dict(enabled=True, mode='signal', beta=-1)))
  with pytest.raises(ValueError, match='teacher_gate.alpha_min'):
    dreamer_agent._validate_teacher_gate_config(
        elements.Config(teacher_gate=dict(
            enabled=True, mode='signal', alpha_min=0.0)))


def test_tg0_configs_parse_and_keep_matched_controls():
  path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(path.read_text())
  defaults = elements.Config(configs['defaults'])

  expected = {
      'tg0_baseline': 'baseline',
      'tg0_signal': 'signal',
      'tg0_shuffle': 'shuffle',
      'tg0_uniform': 'uniform',
  }

  for config_name, mode in expected.items():
    cfg = defaults.update(configs[config_name])
    assert cfg.agent.teacher_gate.enabled is True
    assert cfg.agent.teacher_gate.mode == mode
    assert cfg.agent.teacher_gate.beta == 0.5
    assert cfg.agent.teacher_gate.alpha_min == 0.25
    assert 'teacher_gate' in cfg.logger.filter

def test_imag_loss_source_applies_gate_before_policy_assignment_only():
  source = inspect.getsource(dreamer_agent.imag_loss)
  gate_call = 'policy_loss = _apply_teacher_gate_to_policy_loss(policy_loss, policy_gate)'
  policy_assign = "losses['policy'] = policy_loss"
  value_assign = "losses['value']"

  assert gate_call in source
  assert policy_assign in source
  assert value_assign in source
  assert source.index(gate_call) < source.index(policy_assign)
  assert source.index(policy_assign) < source.index(value_assign)

  repl_source = inspect.getsource(dreamer_agent.repl_loss)
  assert '_apply_teacher_gate_to_policy_loss' not in repl_source

def test_latent_noise_start_timing_uses_noisy_first_feature():
  from pathlib import Path

  text = Path("dreamerv3/agent.py").read_text()
  start = text.index("if timing == 'start':")
  end = text.index("elif timing == 'future_feature':", start)
  block = text[start:end]

  assert "starts_policy, latent_mets = self._apply_latent_noise(starts, active=True)" in block
  assert "first_policy = dict(first)" in block
  assert "for key, value in starts_policy.items():" in block
  assert "first_policy[key] = value[:, None]" in block
  assert "sg(first_policy, skip=self.config.ac_grads), sg(noisy_imgfeat)" in block
  assert "sg(first, skip=self.config.ac_grads), sg(noisy_imgfeat)" not in block


def test_latent_noise_future_feature_timing_keeps_clean_first_and_noises_future():
  from pathlib import Path

  text = Path("dreamerv3/agent.py").read_text()
  start = text.index("elif timing == 'future_feature':")
  end = text.index("else:", start)
  block = text[start:end]

  assert "starts_policy = starts" in block
  assert "noisy_imgfeat, latent_mets = self._apply_latent_noise(" in block
  assert "noisy_imgfeat, active=True)" in block
  assert "sg(first, skip=self.config.ac_grads), sg(noisy_imgfeat)" in block
  assert "first_policy = dict(first)" not in block
  assert "starts_policy, latent_mets = self._apply_latent_noise(starts, active=True)" not in block
