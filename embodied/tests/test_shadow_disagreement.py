import os
import pathlib

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3 import agent as dreamer_agent


REQUIRED_SHADOW_KEYS = {
    'shadow/paired_direction_conflict_weighted_t0_mean',
    'shadow/paired_direction_conflict_t0_mean',
    'shadow/paired_adv_relstd_absdenom_t0_mean',
    'shadow/paired_adv_relstd_absdenom_t0_p95',
    'shadow/paired_mixed_sign_t0_frac',
    'shadow/paired_highimpact_conflict_t0_frac',
    'shadow/paired_return_relstd_absdenom_mean',
    'shadow/paired_return_relstd_absdenom_p95',
    'shadow/paired_latent_absdiff_mean',
    'shadow/main_abs_normalized_advantage_t0_mean',
    'shadow/main_abs_normalized_advantage_mean',
    'shadow/main_abs_normalized_advantage_p95',
}


def _tree_assert_equal(left, right):
  jax.tree.map(
      lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)),
      left, right)


def _tree_all_equal(left, right):
  leaves = jax.tree.leaves(jax.tree.map(
      lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), left, right))
  return all(bool(x) for x in leaves)


def _numpy_direction_stats(adv_t0):
  adv_t0 = np.asarray(adv_t0, np.float32).reshape((3, -1))
  mean_abs_adv = np.abs(adv_t0).mean(0)
  impact_ref = np.maximum(np.median(mean_abs_adv), np.float32(1e-6))
  eps = np.maximum(np.float32(1e-6) * impact_ref, np.float32(1e-8))
  valid = mean_abs_adv > np.float32(0.05) * impact_ref
  raw_conflict = 1 - np.abs(adv_t0.mean(0)) / (mean_abs_adv + eps)
  conflict = np.where(valid, np.clip(raw_conflict, 0, 1), 0)
  impact_weight = mean_abs_adv / (mean_abs_adv + impact_ref)
  relstd = np.where(valid, adv_t0.std(0) / (mean_abs_adv + eps), 0)
  sign_tol = np.float32(0.05) * impact_ref
  mixed_sign = (
      (adv_t0.min(0) < -sign_tol) &
      (adv_t0.max(0) > sign_tol))
  q75 = np.percentile(mean_abs_adv, 75)
  highimpact = mean_abs_adv >= q75
  conflict_high = conflict >= 0.5
  return dict(
      mean_abs_adv=mean_abs_adv,
      impact_ref=impact_ref,
      eps=eps,
      valid=valid,
      conflict=conflict,
      impact_weight=impact_weight,
      relstd=relstd,
      mixed_sign=mixed_sign,
      highimpact=highimpact,
      conflict_high=conflict_high)


def _numpy_metrics(adv, ret, latent, ref_ret, ref_tarval, rscale):
  adv = np.asarray(adv, np.float32).reshape((3, -1, adv.shape[-1]))
  ret = np.asarray(ret, np.float32).reshape((3, -1, ret.shape[-1]))
  latent = np.asarray(latent, np.float32).reshape(
      (3, -1, latent.shape[-2], latent.shape[-1]))
  ref_ret = np.asarray(ref_ret, np.float32).reshape((-1, ref_ret.shape[-1]))
  ref_tarval = np.asarray(ref_tarval, np.float32).reshape(
      (-1, ref_tarval.shape[-1]))
  stats = _numpy_direction_stats(adv[:, :, 0])
  return_mean_abs = np.abs(ret).mean(0)
  return_relstd = ret.std(0) / (return_mean_abs + stats['eps'])
  latent_absdiff = np.abs(latent[1:] - latent[0:1]).mean()
  main_absadv = np.abs(ref_ret - ref_tarval) / np.float32(rscale)
  return {
      'shadow/paired_direction_conflict_weighted_t0_mean':
          (stats['conflict'] * stats['impact_weight']).mean(),
      'shadow/paired_direction_conflict_t0_mean':
          stats['conflict'].mean(),
      'shadow/paired_adv_relstd_absdenom_t0_mean':
          stats['relstd'].mean(),
      'shadow/paired_adv_relstd_absdenom_t0_p95':
          np.percentile(stats['relstd'], 95),
      'shadow/paired_mixed_sign_t0_frac':
          stats['mixed_sign'].astype(np.float32).mean(),
      'shadow/paired_highimpact_conflict_t0_frac':
          (stats['highimpact'] & stats['conflict_high']).astype(
              np.float32).mean(),
      'shadow/paired_return_relstd_absdenom_mean':
          return_relstd.mean(),
      'shadow/paired_return_relstd_absdenom_p95':
          np.percentile(return_relstd, 95),
      'shadow/paired_latent_absdiff_mean':
          latent_absdiff,
      'shadow/main_abs_normalized_advantage_t0_mean':
          main_absadv[:, 0].mean(),
      'shadow/main_abs_normalized_advantage_mean':
          main_absadv.mean(),
      'shadow/main_abs_normalized_advantage_p95':
          np.percentile(main_absadv, 95),
  }


def _manual_inputs(adv):
  adv = np.asarray(adv, np.float32)
  items, horizon = adv.shape[1:]
  ret = np.array([
      [[1.0, 2.0], [2.0, -1.0], [0.0, 3.0], [4.0, 4.0]],
      [[1.5, 2.5], [3.0, -2.0], [2.0, 2.0], [4.0, 5.0]],
      [[0.5, 1.0], [1.0, -3.0], [-2.0, 1.0], [4.0, 6.0]],
  ], np.float32)[:, :items, :horizon]
  latent = np.zeros((3, items, horizon + 1, 2), np.float32)
  latent[1] = 2.0
  latent[2] = -4.0
  ref_ret = ret[0]
  ref_tarval = ref_ret - np.linspace(
      0.1, 0.4, items * horizon, dtype=np.float32).reshape(
          (items, horizon))
  return adv, ret, latent, ref_ret, ref_tarval, np.float32(2.0)


class _FakePred:

  def __init__(self, value, kind='pred'):
    self.value = value
    self.kind = kind

  def pred(self):
    return self.value

  def prob(self, value):
    assert value == 1
    return self.value


class _SpyActionDist:

  def __init__(self, policy, batch):
    self.policy = policy
    self.batch = batch

  def sample(self, seed):
    self.policy.sample_count += 1
    value = jnp.full(
        (self.batch, 1), self.policy.sample_count, jnp.float32)
    if self.policy.randomize:
      _ = jax.random.uniform(seed, value.shape)
    return value


class _SpyPolicy:

  def __init__(self, randomize=False):
    self.randomize = randomize
    self.call_count = 0
    self.sample_count = 0

  def __call__(self, inp, bdims):
    assert bdims == 1
    self.call_count += 1
    return {'action': _SpyActionDist(self, inp.shape[0])}


class _SpyEncoder:

  def __init__(self, randomize=False):
    self.randomize = randomize
    self.calls = 0
    self.trainings = []

  def __call__(self, carry, obs, reset, training=False):
    self.calls += 1
    self.trainings.append(training)
    batch, time = reset.shape
    tokens = jnp.ones((batch, time, 2), jnp.float32)
    if self.randomize:
      tokens = tokens + 0.01 * jax.random.uniform(nj.seed(), tokens.shape)
    return carry, {}, tokens


class _SpyDynamics:

  def __init__(self, randomize=False):
    self.randomize = randomize
    self.observe_calls = 0
    self.observe_trainings = []
    self.imagine_calls = []
    self.last_starts = None

  def observe(self, carry, tokens, action, reset, training=False):
    del action
    self.observe_calls += 1
    self.observe_trainings.append(training)
    batch, time = reset.shape
    offset = jnp.arange(batch * time, dtype=jnp.float32).reshape(
        (batch, time, 1))
    if self.randomize:
      offset = offset + 0.01 * jax.random.uniform(nj.seed(), offset.shape)
    deter = tokens[..., :1] + offset
    stoch = tokens[..., :1, None] * 0 + 0.5
    feat = {'deter': deter, 'stoch': stoch}
    return carry, feat, feat

  def starts(self, entries, carry, nlast):
    del carry
    batch = entries['deter'].shape[0]
    starts = jax.tree.map(
        lambda x: x[:, -nlast:].reshape((batch * nlast, *x.shape[2:])),
        entries)
    self.last_starts = starts
    return starts

  def imagine(self, carry, policy, length, training=False):
    call = {
        'callable': callable(policy),
        'carry': jax.tree.map(lambda x: jnp.array(x), carry),
        'length': length,
        'training': training,
    }
    if not callable(policy):
      call['action_tree'] = jax.tree.map(lambda x: jnp.array(x), policy)
    self.imagine_calls.append(call)
    feat = carry
    feats = []
    actions = []
    for index in range(length):
      if callable(policy):
        action = policy(feat)
      else:
        action = jax.tree.map(lambda x: x[:, index], policy)
      actions.append(action)
      action_value = list(action.values())[0].mean(-1, keepdims=True)
      deter = feat['deter'] + action_value + (index + 1)
      if self.randomize:
        deter = deter + 0.01 * jax.random.uniform(nj.seed(), deter.shape)
      stoch = feat['stoch'] + 0.1
      feat = {'deter': deter, 'stoch': stoch}
      feats.append(feat)
    feats = jax.tree.map(lambda *xs: jnp.stack(xs, 1), *feats)
    actions = jax.tree.map(lambda *xs: jnp.stack(xs, 1), *actions)
    call['returned_features'] = feats
    call['returned_actions'] = actions
    return feat, feats, actions


class _SpyDecoder:

  imgkeys = ('image',)

  def __init__(self, randomize=False):
    self.randomize = randomize
    self.trainings = []

  def __call__(self, carry, feat, reset, training=False):
    del feat
    self.trainings.append(training)
    batch, time = reset.shape
    value = jnp.full((batch, time, 2, 2, 1), 0.5, jnp.float32)
    if self.randomize:
      value = value + 0.25 * jax.random.uniform(nj.seed(), value.shape)
    return carry, {}, {'image': _FakePred(value)}


class _SpyHead:

  def __init__(self, kind):
    self.kind = kind
    self.calls = 0

  def __call__(self, inp, bdims):
    assert bdims == 2
    self.calls += 1
    value = inp.mean(-1)
    if self.kind == 'con':
      value = jnp.full(value.shape, 0.9, jnp.float32)
      return _FakePred(value, 'prob')
    if self.kind == 'slowval':
      value = value + 0.25
    return _FakePred(value)


class _StatsOnlyNorm:

  def __init__(self, offset=0.0, scale=1.0):
    self.offset = jnp.array(offset, jnp.float32)
    self.scale = jnp.array(scale, jnp.float32)
    self.stats_calls = 0

  def stats(self):
    self.stats_calls += 1
    return self.offset, self.scale

  def __call__(self, *args, **kwargs):
    raise AssertionError('shadow report must not call normalizer __call__')


def _feat2tensor(feat):
  return jnp.concatenate([
      feat['deter'],
      feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))], -1)


def _spy_agent(enabled=True, randomize=False):
  agent = object.__new__(dreamer_agent.Agent)
  agent.config = elements.Config(
      report=True,
      report_gradnorms=False,
      imag_last=2,
      imag_length=3,
      imag_loss=dict(slowtar=False, lam=0.95),
      contdisc=True,
      horizon=333,
      shadow_disagreement=dict(enabled=enabled, views=3))
  agent.scales = {}
  agent.act_space = {'action': None}
  agent.feat2tensor = _feat2tensor
  agent.enc = _SpyEncoder(randomize)
  agent.dyn = _SpyDynamics(randomize)
  agent.dec = _SpyDecoder(randomize)
  agent.pol = _SpyPolicy(randomize)
  agent.rew = _SpyHead('rew')
  agent.con = _SpyHead('con')
  agent.val = _SpyHead('val')
  agent.slowval = _SpyHead('slowval')
  agent.valnorm = _StatsOnlyNorm(0.0, 1.0)
  agent.retnorm = _StatsOnlyNorm(0.0, 2.0)
  agent.advnorm = _StatsOnlyNorm()

  def apply_replay_context(carry, data):
    obs = {k: data[k] for k in (
        'image', 'is_first', 'is_last', 'is_terminal', 'reward')}
    prevact = {'action': data['action']}
    return carry, obs, prevact, data['stepid']

  def loss(carry, obs, prevact, training):
    del obs, prevact, training
    if randomize:
      token_base = jax.random.uniform(nj.seed(), (2, 4, 2))
    else:
      token_base = jnp.ones((2, 4, 2), jnp.float32)
    return 0.0, (carry, None, {'tokens': token_base}, {
        'loss/probe': jnp.array(3.0, jnp.float32)})

  agent._apply_replay_context = apply_replay_context
  agent.loss = loss
  return agent


def _report_inputs():
  data = {
      'image': jnp.arange(2 * 4 * 2 * 2, dtype=jnp.uint8).reshape(
          (2, 4, 2, 2, 1)),
      'is_first': jnp.array(
          [[True, False, False, False], [True, False, False, False]]),
      'is_last': jnp.zeros((2, 4), bool),
      'is_terminal': jnp.zeros((2, 4), bool),
      'reward': jnp.zeros((2, 4), jnp.float32),
      'action': jnp.zeros((2, 4, 1), jnp.float32),
      'stepid': jnp.zeros((2, 4, 20), jnp.uint8),
  }
  carry = (
      {},
      {'deter': jnp.zeros((2, 1), jnp.float32),
       'stoch': jnp.zeros((2, 1, 1), jnp.float32)},
      {})
  return carry, data


def _run_report_with_seed(enabled, preconsume=False):
  agent = _spy_agent(enabled=enabled, randomize=True)
  carry, data = _report_inputs()

  def report(carry, data):
    if preconsume:
      _ = nj.seed()
    return dreamer_agent.Agent.report(agent, carry, data)

  _, out = nj.pure(report)(
      {}, carry, data, seed=jnp.array([123, 456], jnp.uint32))
  return agent, out


class TestShadowDisagreement:

  def test_hand_expected_primary_arithmetic_not_denominator_weighted(self):
    adv = np.array([
        [[1.0], [10.0]],
        [[1.0], [-10.0]],
        [[1.0], [-10.0]],
    ], np.float32)
    ret = np.ones((3, 2, 1), np.float32)
    latent = np.zeros((3, 2, 2, 1), np.float32)
    ref_ret = np.ones((2, 1), np.float32)
    ref_tarval = np.zeros((2, 1), np.float32)

    metrics = dreamer_agent._shadow_disagreement_metrics(
        adv, ret, latent, ref_ret, ref_tarval, np.float32(1.0))

    mean_abs_adv = np.array([1.0, 10.0], np.float32)
    impact_ref = np.float32(5.5)
    eps = np.float32(5.5e-6)
    conflict0 = (
        np.float32(1.0) - np.float32(1.0) / (mean_abs_adv[0] + eps))
    conflict1 = (
        np.float32(1.0) - np.float32(10.0 / 3.0) /
        (mean_abs_adv[1] + eps))
    impact_weight0 = mean_abs_adv[0] / (mean_abs_adv[0] + impact_ref)
    impact_weight1 = mean_abs_adv[1] / (mean_abs_adv[1] + impact_ref)
    expected = (
        conflict0 * impact_weight0 + conflict1 * impact_weight1) / 2
    denominator_weighted = (
        conflict0 * impact_weight0 + conflict1 * impact_weight1) / (
            impact_weight0 + impact_weight1)
    actual = float(metrics[
        'shadow/paired_direction_conflict_weighted_t0_mean'])
    np.testing.assert_allclose(expected, np.float32(0.21505424), rtol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert not np.isclose(actual, denominator_weighted)

  def test_hand_expected_impact_eps_valid_and_same_sign_residue(self):
    stats = dreamer_agent._shadow_direction_stats(np.array([
        [2.0, 10.0, 1.0],
        [2.0, 20.0, 1.0],
        [2.0, 30.0, 1.0],
    ], np.float32))

    np.testing.assert_allclose(stats['mean_abs_adv'], [2.0, 20.0, 1.0])
    np.testing.assert_allclose(stats['impact_ref'], 2.0)
    np.testing.assert_allclose(stats['eps'], 2e-6)
    np.testing.assert_array_equal(stats['valid'], [True, True, True])
    # Exact IEEE float32 results of the registered guarded formula.
    expected_conflict0 = np.float32(9.5367431640625e-7)
    expected_conflict1 = np.float32(1.1920928955078125e-7)
    np.testing.assert_array_equal(
        np.asarray(stats['conflict'][0]), expected_conflict0)
    np.testing.assert_array_equal(
        np.asarray(stats['conflict'][1]), expected_conflict1)
    assert float(stats['conflict'][0]) > 0.0
    assert float(stats['conflict'][0]) < 1e-5

  def test_hand_expected_strict_valid_boundary_and_all_zero_mask(self):
    boundary = dreamer_agent._shadow_direction_stats(np.array([
        [20.0, 1.0],
        [20.0, 1.0],
        [20.0, 1.0],
    ], np.float32))
    np.testing.assert_allclose(boundary['impact_ref'], 10.5)
    np.testing.assert_array_equal(boundary['valid'], [True, True])

    # Choose an exactly representable float32 boundary:
    # impact_ref=20 and 0.05*impact_ref=1 exactly.
    strict = dreamer_agent._shadow_direction_stats(np.array([
        [20.0, 1.0, 20.0],
        [20.0, 1.0, 20.0],
        [20.0, 1.0, 20.0],
    ], np.float32))
    np.testing.assert_allclose(
        strict['mean_abs_adv'], [20.0, 1.0, 20.0])
    np.testing.assert_allclose(strict['impact_ref'], 20.0)
    np.testing.assert_array_equal(strict['valid'], [True, False, True])

    adv, ret, latent, ref_ret, ref_tarval, rscale = _manual_inputs(
        np.zeros((3, 2, 1), np.float32))
    metrics = dreamer_agent._shadow_disagreement_metrics(
        adv, ret[:, :2, :1], latent[:, :2, :2], ref_ret[:2, :1],
        ref_tarval[:2, :1], rscale)
    assert float(metrics[
        'shadow/paired_direction_conflict_weighted_t0_mean']) == 0.0
    assert float(metrics['shadow/paired_direction_conflict_t0_mean']) == 0.0

  def test_hand_expected_sign_tol_q75_and_highimpact_fraction(self):
    adv_t0 = np.array([
        [100.0, -100.0, -10.0, 0.10],
        [-100.0, 100.0, 10.0, 0.10],
        [100.0, 100.0, 10.0, 0.10],
    ], np.float32)
    stats = dreamer_agent._shadow_direction_stats(adv_t0)

    np.testing.assert_allclose(
        stats['mean_abs_adv'], [100.0, 100.0, 10.0, 0.1])
    np.testing.assert_allclose(stats['impact_ref'], 55.0, rtol=1e-6)
    sign_tol = 2.75
    assert adv_t0[:, 2].min() < -sign_tol
    assert adv_t0[:, 3].min() > -sign_tol
    np.testing.assert_array_equal(
        stats['mixed_sign'], [True, True, True, False])
    np.testing.assert_array_equal(
        stats['highimpact'], [True, True, False, False])
    np.testing.assert_array_equal(
        stats['conflict_high'], [True, True, True, False])
    np.testing.assert_allclose(
        np.asarray(stats['highimpact'] & stats['conflict_high']).mean(), 0.5)

  def test_hand_expected_latent_auxiliary_vs_reference_difference(self):
    adv = np.ones((3, 1, 1), np.float32)
    ret = np.ones((3, 1, 1), np.float32)
    latent = np.array([
        [[[1.0, 3.0], [5.0, 7.0]]],
        [[[2.0, 4.0], [6.0, 8.0]]],
        [[[4.0, 7.0], [10.0, 13.0]]],
    ], np.float32)
    metrics = dreamer_agent._shadow_disagreement_metrics(
        adv, ret, latent, ret[0], np.zeros((1, 1), np.float32), 1.0)
    expected = np.mean([
        1.0, 1.0, 1.0, 1.0,
        3.0, 4.0, 5.0, 6.0,
    ])
    np.testing.assert_allclose(
        metrics['shadow/paired_latent_absdiff_mean'], expected)

  @pytest.mark.parametrize('value', [2.0, -2.0])
  def test_equal_same_sign_uses_registered_formula_residue(self, value):
    adv, ret, latent, ref_ret, ref_tarval, rscale = _manual_inputs(
        np.full((3, 1, 1), value, np.float32))
    metrics = dreamer_agent._shadow_disagreement_metrics(
        adv, ret, latent, ref_ret, ref_tarval, rscale)
    expected = _numpy_metrics(adv, ret, latent, ref_ret, ref_tarval, rscale)

    actual = float(metrics[
        'shadow/paired_direction_conflict_weighted_t0_mean'])
    assert actual > 0.0
    assert actual < 1e-5
    np.testing.assert_allclose(actual, expected[
        'shadow/paired_direction_conflict_weighted_t0_mean'], rtol=1e-6)

  def test_supplementary_numpy_expected_values_and_required_names(self):
    adv, ret, latent, ref_ret, ref_tarval, rscale = _manual_inputs(np.array([
        [[1.0, 0.5], [-2.0, -1.0], [2.0, 0.1], [0.0, 0.0]],
        [[2.0, 0.1], [-4.0, -1.0], [-2.0, 0.2], [0.0, 0.0]],
        [[4.0, 0.2], [-1.0, -0.5], [1.0, 0.3], [0.0, 0.0]],
    ], np.float32))
    expected = _numpy_metrics(adv, ret, latent, ref_ret, ref_tarval, rscale)

    metrics = dreamer_agent._shadow_disagreement_metrics(
        adv, ret, latent, ref_ret, ref_tarval, rscale)

    assert set(metrics) == REQUIRED_SHADOW_KEYS
    for key, value in metrics.items():
      value = np.asarray(value)
      assert value.shape == (), key
      assert np.isfinite(value), key
      np.testing.assert_allclose(value, expected[key], rtol=1e-6, atol=1e-6)

  @pytest.mark.parametrize('bad_views', [True, False, '3', 3.0, 2, 4])
  def test_view_count_config_validation_is_exact(self, bad_views):
    with pytest.raises(ValueError, match='exactly 3'):
      dreamer_agent._validate_shadow_disagreement_config(
          elements.Config(shadow_disagreement=dict(
              enabled=True, views=bad_views)))

  def test_metric_shape_validation_is_exact(self):
    adv, ret, latent, ref_ret, ref_tarval, rscale = _manual_inputs(
        np.ones((3, 2, 1), np.float32))
    with pytest.raises(ValueError, match='adv_views'):
      dreamer_agent._shadow_disagreement_metrics(
          adv[:2], ret, latent, ref_ret, ref_tarval, rscale)
    with pytest.raises(ValueError, match='ret_views'):
      dreamer_agent._shadow_disagreement_metrics(
          adv, ret[:2], latent, ref_ret, ref_tarval, rscale)
    with pytest.raises(ValueError, match='latent_views'):
      dreamer_agent._shadow_disagreement_metrics(
          adv, ret, latent[:2], ref_ret, ref_tarval, rscale)
    with pytest.raises(ValueError, match='incompatible'):
      dreamer_agent._shadow_disagreement_metrics(
          adv, ret[:, :, :0], latent, ref_ret, ref_tarval, rscale)

  def test_shadow_config_parses_and_sets_logger(self):
    path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(path.read_text())
    defaults = elements.Config(configs['defaults'])
    enabled = defaults.update(configs['shadow_disagreement'])

    assert defaults.agent.shadow_disagreement.enabled is False
    assert defaults.agent.shadow_disagreement.views == 3
    assert enabled.agent.shadow_disagreement.enabled is True
    assert enabled.agent.shadow_disagreement.views == 3
    assert enabled.run.report_every == 30
    assert enabled.run.report_batches == 1
    for term in ('score', 'length', 'fps', 'ratio', 'train/loss/',
                 'train/rand/', 'shadow/'):
      assert term in enabled.logger.filter

  def test_real_shadow_rollout_contract(self):
    agent = _spy_agent(enabled=True, randomize=True)
    carry, data = _report_inputs()
    _, obs, prevact, _ = agent._apply_replay_context(carry, data)

    def report(carry, obs, prevact):
      return dreamer_agent.Agent._shadow_disagreement_report(
          agent, carry, obs, prevact)

    _, metrics = nj.pure(report)(
        {}, carry, obs, prevact, seed=jnp.array([7, 8], jnp.uint32))

    assert agent.enc.calls == 1
    assert agent.dyn.observe_calls == 1
    assert agent.enc.trainings == [False]
    assert agent.dyn.observe_trainings == [False]
    assert len(agent.dyn.imagine_calls) == 3
    ref, aux1, aux2 = agent.dyn.imagine_calls
    assert ref['callable']
    assert not aux1['callable']
    assert not aux2['callable']
    assert ref['training'] is False
    assert aux1['training'] is False
    assert aux2['training'] is False
    assert ref['length'] == agent.config.imag_length
    assert aux1['length'] == agent.config.imag_length
    assert aux2['length'] == agent.config.imag_length
    assert agent.pol.sample_count == agent.config.imag_length
    _tree_assert_equal(aux1['action_tree'], ref['returned_actions'])
    _tree_assert_equal(aux2['action_tree'], ref['returned_actions'])
    _tree_assert_equal(aux1['carry'], agent.dyn.last_starts)
    _tree_assert_equal(aux2['carry'], agent.dyn.last_starts)
    assert not _tree_all_equal(ref['returned_features'], aux1[
        'returned_features'])
    assert not _tree_all_equal(ref['returned_features'], aux2[
        'returned_features'])
    assert not _tree_all_equal(aux1['returned_features'], aux2[
        'returned_features'])
    assert aux1['action_tree']['action'].shape[1] == agent.config.imag_length
    assert aux2['action_tree']['action'].shape[1] == agent.config.imag_length
    assert all(x is False for x in agent.dec.trainings)
    assert agent.valnorm.stats_calls == 1
    assert agent.retnorm.stats_calls == 1
    assert set(metrics) == REQUIRED_SHADOW_KEYS
    for key, value in metrics.items():
      value = np.asarray(value)
      assert value.shape == (), key
      assert np.isfinite(value), key
    assert float(metrics['shadow/paired_latent_absdiff_mean']) > 0

  def test_report_rng_order_negative_control_changes_openloop(self):
    _, (_, baseline) = _run_report_with_seed(False)
    _, (_, shifted) = _run_report_with_seed(False, preconsume=True)

    assert not np.array_equal(
        np.asarray(baseline['openloop/image']),
        np.asarray(shifted['openloop/image']))

  def test_report_rng_order_preserves_original_outputs(self):
    disabled_agent, (disabled_carry, disabled) = _run_report_with_seed(False)
    enabled_agent, (enabled_carry, enabled) = _run_report_with_seed(True)

    assert 'loss/probe' in disabled
    assert set(enabled) == set(disabled) | REQUIRED_SHADOW_KEYS
    for key, value in disabled.items():
      np.testing.assert_array_equal(np.asarray(value), np.asarray(enabled[key]))
    _tree_assert_equal(disabled_carry, enabled_carry)
    assert disabled_agent.dyn.imagine_calls
    assert len(enabled_agent.dyn.imagine_calls) == (
        len(disabled_agent.dyn.imagine_calls) + 3)
    for key in REQUIRED_SHADOW_KEYS:
      value = np.asarray(enabled[key])
      assert value.shape == (), key
      assert np.isfinite(value), key

  def test_opt_in_tiny_agent_fixed_batch_invariance(self):
    if os.environ.get('COMP9991_RUN_SHADOW_TINY_AGENT') != '1':
      pytest.skip(
          'Set COMP9991_RUN_SHADOW_TINY_AGENT=1 on Katana to run this '
          'expensive fixed-batch invariance check.')

    path = pathlib.Path(__file__).parents[2] / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(path.read_text())
    base = elements.Config(configs['defaults']).update(configs['debug'])
    obs_space = {
        'vector': elements.Space(np.float32, (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    act_space = {'action': elements.Space(np.float32, (2,), -1, 1)}

    def make_model(enabled):
      values = dict(base.agent)
      values.update(
          seed=0,
          batch_size=2,
          batch_length=5,
          replay_context=1,
          report_length=5,
          replica=0,
          replicas=1,
          logdir='')
      cfg = elements.Config(values)
      cfg = cfg.update(
          shadow_disagreement=dict(enabled=enabled, views=3),
          imag_length=2,
          imag_last=2)
      model = object.__new__(dreamer_agent.Agent)
      dreamer_agent.Agent.__init__(model, obs_space, act_space, cfg)
      return model

    batch = 2
    length = 5
    data = {
        'vector': jnp.arange(batch * length * 3, dtype=jnp.float32).reshape(
            (batch, length, 3)) / 10.0,
        'reward': jnp.zeros((batch, length), jnp.float32),
        'is_first': jnp.array(
            [[True, False, False, False, False],
             [True, False, False, False, False]]),
        'is_last': jnp.zeros((batch, length), bool),
        'is_terminal': jnp.zeros((batch, length), bool),
        'action': jnp.zeros((batch, length, 2), jnp.float32),
        'stepid': jnp.zeros((batch, length, 20), jnp.uint8),
    }
    seed = jnp.array([11, 22], jnp.uint32)

    def full_data_for(model):
      full_data = dict(data)
      for key, space in model.ext_space.items():
        if key not in full_data:
          full_data[key] = jnp.zeros(
              (batch, length, *tuple(space.shape)), space.dtype)
      return full_data

    def loss_snapshot(enabled):
      model = make_model(enabled)
      full_data = full_data_for(model)
      state, carry = nj.pure(model.init_train)(
          {}, batch, seed=seed, create=True)

      def collect(carry, data):
        carry3, obs, prevact, _ = model._apply_replay_context(carry, data)
        _, (_, _, loss_outs, loss_metrics) = model.loss(
            carry3, obs, prevact, training=False)
        enc_carry, dyn_carry, _ = carry3
        reset = obs['is_first']
        batch_size, time = reset.shape
        _, _, tokens = model.enc(
            enc_carry, obs, reset, training=False)
        dyn_carry, dyn_entries, _, repfeat, _ = model.dyn.loss(
            dyn_carry, tokens, prevact, reset, training=False)
        k = min(model.config.imag_last or time, time)
        horizon = model.config.imag_length
        starts = model.dyn.starts(dyn_entries, dyn_carry, k)
        policyfn = lambda feat: dreamer_agent.sample(
            model.pol(model.feat2tensor(feat), 1))
        _, imgfeat, imgprevact = model.dyn.imagine(
            starts, policyfn, horizon, training=False)
        first = jax.tree.map(
            lambda x: x[:, -k:].reshape(
                (batch_size * k, 1, *x.shape[2:])), repfeat)
        imgfeat = dreamer_agent.concat([
            dreamer_agent.sg(first, skip=model.config.ac_grads),
            dreamer_agent.sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = dreamer_agent.concat([imgprevact, lastact], 1)
        inp = model.feat2tensor(imgfeat)
        pred_rew = model.rew(inp, 2).pred()
        pred_con = model.con(inp, 2).prob(1)
        _, imgloss_out, _ = dreamer_agent.imag_loss(
            imgact,
            pred_rew,
            pred_con,
            model.pol(inp, 2),
            model.val(inp, 2),
            model.slowval(inp, 2),
            model.retnorm, model.valnorm, model.advnorm,
            update=False,
            contdisc=model.config.contdisc,
            horizon=model.config.horizon,
            **model.config.imag_loss)
        voffset, vscale = model.valnorm.stats()
        val = model.val(inp, 2).pred() * vscale + voffset
        slowval = model.slowval(inp, 2).pred() * vscale + voffset
        tarval = slowval if model.config.imag_loss.slowtar else val
        _, rscale = model.retnorm(imgloss_out['ret'], False)
        return {
            'posterior_repfeat': loss_outs['repfeat'],
            'main_imagined_transition_actions': imgprevact,
            'main_imagined_final_action': lastact,
            'main_imagined_latent_features': imgfeat,
            'main_predicted_rewards': pred_rew,
            'main_predicted_continuations': pred_con,
            'main_lambda_returns': imgloss_out['ret'],
            'main_advantages': (
                imgloss_out['ret'] - tarval[:, :-1]) / rscale,
            'losses': loss_outs['losses'],
            'metrics': loss_metrics,
        }

      _, snapshot = nj.pure(collect)(
          state, carry, full_data, seed=seed, create=True)
      return snapshot

    def run(enabled):
      model = make_model(enabled)
      full_data = full_data_for(model)
      state, carry = nj.pure(model.init_train)(
          {}, batch, seed=seed, create=True)
      state, train_out = nj.pure(model.train)(
          state, carry, full_data, seed=seed, create=True)
      train_carry, train_outs, train_metrics = train_out
      state, report_out = nj.pure(model.report)(
          state, train_carry, full_data, seed=seed, create=False)
      return state, train_carry, train_outs, train_metrics, report_out

    disabled_snapshot = loss_snapshot(False)
    enabled_snapshot = loss_snapshot(True)
    _tree_assert_equal(disabled_snapshot, enabled_snapshot)

    disabled = run(False)
    enabled = run(True)

    for left, right in zip(disabled[:4], enabled[:4]):
      _tree_assert_equal(left, right)
    assert 'replay' in disabled[2]
    assert 'replay' in enabled[2]
    _tree_assert_equal(disabled[2]['replay'], enabled[2]['replay'])
    disabled_report_carry, disabled_report = disabled[4]
    enabled_report_carry, enabled_report = enabled[4]
    _tree_assert_equal(disabled_report_carry, enabled_report_carry)
    assert set(enabled_report) == set(disabled_report) | REQUIRED_SHADOW_KEYS
    for key in disabled_report:
      np.testing.assert_array_equal(
          np.asarray(disabled_report[key]), np.asarray(enabled_report[key]))
    for key in REQUIRED_SHADOW_KEYS:
      value = np.asarray(enabled_report[key])
      assert value.shape == (), key
      assert np.isfinite(value), key
    assert float(enabled_report['shadow/paired_latent_absdiff_mean']) > 0
