import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    _validate_shadow_disagreement_config(config)
    _validate_teacher_gate_config(config)
    _validate_latent_noise_config(config)

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    latent_config = self.config.get('latent_noise', {})
    latent_policy_active = (
        training and
        latent_config.get('enabled', False) and
        latent_config.get('apply_train_policy', True))
    use_latent_policy = (
        latent_policy_active and
        float(latent_config.get('prob', 0.0)) > 0.0 and
        float(latent_config.get('sigma', 0.0)) > 0.0)
    if use_latent_policy:
      policy_gate = None
    else:
      policy_gate, gate_mets = self._teacher_gate_policy_gate(
          starts, imgfeat, imgprevact, H, training)
      metrics.update(prefix(gate_mets, 'teacher_gate'))
      _, latent_mets = self._apply_latent_noise(
          starts, active=latent_policy_active)
      metrics.update(prefix(latent_mets, 'latent_noise'))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        policy_gate=policy_gate,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    if use_latent_policy:
      timing = self.config.agent.latent_noise.timing
      if timing == 'start':
        starts_policy, latent_mets = self._apply_latent_noise(starts, active=True)
        metrics.update(prefix(latent_mets, 'latent_noise'))
        _, noisy_imgfeat, noisy_imgprevact = self.dyn.imagine(
            starts_policy, policyfn, H, training)
        first_policy = dict(first)
        for key, value in starts_policy.items():
          if key in first_policy:
            first_policy[key] = value[:, None]
        noisy_imgfeat = concat([
            sg(first_policy, skip=self.config.ac_grads), sg(noisy_imgfeat)], 1)
      elif timing == 'future_feature':
        starts_policy = starts
        _, noisy_imgfeat, noisy_imgprevact = self.dyn.imagine(
            starts_policy, policyfn, H, training)
        noisy_imgfeat, latent_mets = self._apply_latent_noise(
            noisy_imgfeat, active=True)
        metrics.update(prefix(latent_mets, 'latent_noise'))
        noisy_imgfeat = concat([
            sg(first, skip=self.config.ac_grads), sg(noisy_imgfeat)], 1)
      else:
        raise ValueError(f'Unknown latent_noise timing: {timing}')
      noisy_lastact = policyfn(jax.tree.map(lambda x: x[:, -1], noisy_imgfeat))
      noisy_lastact = jax.tree.map(lambda x: x[:, None], noisy_lastact)
      noisy_imgact = concat([noisy_imgprevact, noisy_lastact], 1)
      assert all(
          x.shape[:2] == (B * K, H + 1)
          for x in jax.tree.leaves(noisy_imgfeat))
      assert all(
          x.shape[:2] == (B * K, H + 1)
          for x in jax.tree.leaves(noisy_imgact))
      policy_gate, gate_mets = self._teacher_gate_policy_gate(
          starts_policy, noisy_imgfeat, noisy_imgprevact, H, training)
      metrics.update(prefix(gate_mets, 'teacher_gate'))
      noisy_inp = self.feat2tensor(noisy_imgfeat)
      noisy_los, _, _ = imag_loss(
          noisy_imgact,
          self.rew(noisy_inp, 2).pred(),
          self.con(noisy_inp, 2).prob(1),
          self.pol(noisy_inp, 2),
          self.val(noisy_inp, 2),
          self.slowval(noisy_inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=False,
          policy_gate=policy_gate,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses['policy'] = noisy_los['policy'].mean(1).reshape((B, K))

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    metrics.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    shadow_config = self.config.get('shadow_disagreement', {})
    if shadow_config.get('enabled', False):
      metrics.update(self._shadow_disagreement_report(carry, obs, prevact))

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _shadow_disagreement_report(self, carry, obs, prevact):
    (_, dyn_carry, _) = carry
    reset = obs['is_first']
    B, T = reset.shape
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length

    _, _, tokens = self.enc(carry[0], obs, reset, training=False)
    dyn_carry, dyn_entries, repfeat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    latent_config = self.config.get('latent_noise', {})
    starts, latent_mets = self._apply_latent_noise(
        starts,
        active=(
            latent_config.get('enabled', False) and
            latent_config.get('apply_report_shadow', True)))
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    first = sg(first)

    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, reffeat, refprevact = self.dyn.imagine(
        starts, policyfn, H, training=False)
    reffeat = concat([first, sg(reffeat)], 1)

    auxfeats = []
    common = sg(refprevact)
    for _ in range(2):
      _, auxfeat, _ = self.dyn.imagine(
          starts, common, H, training=False)
      auxfeats.append(concat([first, sg(auxfeat)], 1))

    views = [reffeat, *auxfeats]
    voffset, vscale = self.valnorm.stats()
    _, frozen_rscale = self.retnorm.stats()
    stats = [
        self._shadow_view_stats(view, voffset, vscale, frozen_rscale)
        for view in views]
    adv_views = jnp.stack([x['adv'] for x in stats], 0)
    ret_views = jnp.stack([x['ret'] for x in stats], 0)
    latent_views = jnp.stack([
        sg(self.feat2tensor(view)) for view in views], 0)
    metrics = _shadow_disagreement_metrics(
        sg(adv_views),
        sg(ret_views),
        sg(latent_views),
        sg(stats[0]['ret']),
        sg(stats[0]['tarval']),
        sg(frozen_rscale))
    if (latent_config.get('enabled', False) and
        latent_config.get('apply_report_shadow', True)):
      metrics.update(prefix(latent_mets, 'latent_noise'))
    return metrics

  def _shadow_view_stats(self, imgfeat, voffset, vscale, frozen_rscale):
    inp = self.feat2tensor(imgfeat)
    rew = self.rew(inp, 2).pred()
    con = self.con(inp, 2).prob(1)
    val = self.val(inp, 2).pred() * vscale + voffset
    slowval = self.slowval(inp, 2).pred() * vscale + voffset
    tarval = slowval if self.config.imag_loss.slowtar else val
    disc = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    last = jnp.zeros_like(con)
    term = 1 - con
    ret = lambda_return(
        last, term, rew, tarval, tarval, disc, self.config.imag_loss.lam)
    tarval = tarval[:, :-1]
    adv = (ret - tarval) / sg(frozen_rscale)
    return {'ret': ret, 'tarval': tarval, 'adv': adv}


  def _apply_latent_noise(self, starts, active=True):
    config = self.config.get('latent_noise', {})
    enabled = bool(config.get('enabled', False)) and bool(active)
    prob = float(config.get('prob', 0.0))
    sigma = float(config.get('sigma', 0.0))
    metrics = {
        'enabled': f32(enabled),
        'prob': f32(prob),
        'sigma': f32(sigma),
        'applied_frac': f32(0.0),
        'deter_absdiff_mean': f32(0.0),
        'deter_rel_rms_mean': f32(0.0),
    }
    if not enabled or prob <= 0.0 or sigma <= 0.0:
      return dict(starts), metrics

    eps = float(config.get('eps', 1e-6))
    stop_grad_noise = bool(config.get('stop_grad_noise', True))
    deter = starts['deter']
    base = sg(deter) if stop_grad_noise else deter
    rms = jnp.sqrt(jnp.mean(f32(base) ** 2, axis=-1, keepdims=True)) + f32(eps)
    mask = jax.random.bernoulli(
        nj.seed(), f32(prob), deter.shape[:-1] + (1,))
    mask = mask.astype(deter.dtype)
    noise = jax.random.normal(nj.seed(), deter.shape, dtype=f32)
    noise = noise * f32(sigma) * rms
    if stop_grad_noise:
      mask = sg(mask)
      noise = sg(noise)
    delta = f32(mask) * noise
    noisy = dict(starts)
    noisy['deter'] = (f32(deter) + delta).astype(deter.dtype)
    metrics = {
        'enabled': f32(1.0),
        'prob': f32(prob),
        'sigma': f32(sigma),
        'applied_frac': f32(mask).mean(),
        'deter_absdiff_mean': jnp.abs(delta).mean(),
        'deter_rel_rms_mean': (jnp.abs(delta) / rms).mean(),
    }
    return noisy, metrics


  def _teacher_gate_policy_gate(self, starts, ref_imgfeat, refprevact, H, training):
    gate_config = self.config.get('teacher_gate', {})
    if not gate_config.get('enabled', False):
      return None, {}

    mode = str(gate_config.get('mode', 'baseline')).lower()
    beta = float(gate_config.get('beta', 0.5))
    alpha_min = float(gate_config.get('alpha_min', 0.25))
    eps = float(gate_config.get('eps', 1e-6))

    ref_leaf = jax.tree.leaves(ref_imgfeat)[0]
    num_starts = ref_leaf.shape[0]

    if mode in ('disabled', 'none'):
      return None, {}

    if mode == 'baseline':
      alpha = jnp.ones((num_starts,), f32)
      conflict_weighted = jnp.zeros((num_starts,), f32)
    else:
      first = jax.tree.map(lambda x: x[:, :1], ref_imgfeat)
      common = sg(refprevact)
      auxfeats = []
      for _ in range(2):
        _, auxfeat, _ = self.dyn.imagine(starts, common, H, training)
        auxfeats.append(concat([first, sg(auxfeat)], 1))

      views = [sg(ref_imgfeat), *auxfeats]
      voffset, vscale = self.valnorm.stats()
      _, frozen_rscale = self.retnorm.stats()
      stats = [
          self._shadow_view_stats(view, voffset, vscale, frozen_rscale)
          for view in views]
      adv_views = jnp.stack([x['adv'] for x in stats], 0)
      direction = _shadow_direction_stats(sg(adv_views)[:, :, 0])
      conflict_weighted = sg(direction['conflict'] * direction['impact_weight'])
      alpha = _teacher_gate_alpha_from_conflict(
          conflict_weighted, beta=beta, alpha_min=alpha_min, eps=eps)

      if mode == 'shuffle':
        alpha = jax.random.permutation(nj.seed(), alpha, axis=0)
      elif mode == 'uniform':
        alpha = jnp.ones_like(alpha) * sg(alpha.mean())
      elif mode == 'signal':
        pass
      else:
        raise ValueError(f'Unknown teacher_gate.mode: {mode!r}.')

    alpha = sg(jnp.clip(alpha, f32(alpha_min), f32(1.0)))
    gate = alpha[:, None]

    metrics = {
        'enabled': f32(1.0),
        'mode_code': f32(_teacher_gate_mode_code(mode)),
        'alpha_mean': alpha.mean(),
        'alpha_min': alpha.min(),
        'alpha_max': alpha.max(),
        'alpha_std': alpha.std(),
        'active_frac': jnp.asarray(alpha < f32(0.999), f32).mean(),
        'conflict_weighted_mean': conflict_weighted.mean(),
        'conflict_weighted_max': conflict_weighted.max(),
    }
    return gate, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def _validate_shadow_disagreement_config(config):
  shadow = config.get('shadow_disagreement', {})
  views = shadow.get('views', 3)
  if isinstance(views, bool) or not isinstance(views, (int, np.integer)) or (
      views != 3):
    raise ValueError(
        f'shadow_disagreement.views must be exactly 3, got {views}.')




def _validate_teacher_gate_config(config):
  gate = config.get('teacher_gate', {})
  if not gate:
    return
  mode = str(gate.get('mode', 'disabled')).lower()
  allowed = ('disabled', 'none', 'baseline', 'signal', 'shuffle', 'uniform')
  if mode not in allowed:
    raise ValueError(
        f'teacher_gate.mode must be one of {allowed}, got {mode!r}.')
  beta = float(gate.get('beta', 0.5))
  alpha_min = float(gate.get('alpha_min', 0.25))
  eps = float(gate.get('eps', 1e-6))
  if not np.isfinite(beta) or beta < 0:
    raise ValueError(
        f'teacher_gate.beta must be finite and nonnegative, got {beta}.')
  if not np.isfinite(alpha_min) or not 0.0 < alpha_min <= 1.0:
    raise ValueError(
        f'teacher_gate.alpha_min must be in (0, 1], got {alpha_min}.')
  if not np.isfinite(eps) or eps <= 0:
    raise ValueError(
        f'teacher_gate.eps must be finite and positive, got {eps}.')


def _validate_latent_noise_config(config):
  if config.agent.latent_noise.timing not in ('start', 'future_feature'):
    raise ValueError(
        'agent.latent_noise.timing must be start or future_feature, '
        f'got {config.agent.latent_noise.timing!r}')
  latent = config.get('latent_noise', {})
  if not latent:
    return
  target = str(latent.get('target', 'deter'))
  if target != 'deter':
    raise ValueError(
        f'latent_noise.target must be "deter", got {target!r}.')
  scale = str(latent.get('scale', 'rms'))
  if scale != 'rms':
    raise ValueError(
        f'latent_noise.scale must be "rms", got {scale!r}.')

  def get_float(name, default):
    value = latent.get(name, default)
    if isinstance(value, (bool, np.bool_)):
      raise ValueError(f'latent_noise.{name} must be numeric, got {value}.')
    try:
      return float(value)
    except (TypeError, ValueError) as exc:
      raise ValueError(
          f'latent_noise.{name} must be numeric, got {value!r}.') from exc

  prob = get_float('prob', 0.0)
  sigma = get_float('sigma', 0.0)
  eps = get_float('eps', 1e-6)
  if not np.isfinite(prob) or not 0.0 <= prob <= 1.0:
    raise ValueError(
        f'latent_noise.prob must be finite and in [0, 1], got {prob}.')
  if not np.isfinite(sigma) or sigma < 0:
    raise ValueError(
        f'latent_noise.sigma must be finite and nonnegative, got {sigma}.')
  if not np.isfinite(eps) or eps <= 0:
    raise ValueError(
        f'latent_noise.eps must be finite and positive, got {eps}.')

  bool_fields = (
      'stop_grad_noise',
      'apply_train_policy',
      'apply_train_value',
      'apply_report_shadow',
      'apply_policy_eval',
      'apply_env_acting',
  )
  for name in bool_fields:
    value = latent.get(name, {
        'stop_grad_noise': True,
        'apply_train_policy': True,
        'apply_train_value': False,
        'apply_report_shadow': True,
        'apply_policy_eval': False,
        'apply_env_acting': False,
    }[name])
    if not isinstance(value, (bool, np.bool_)):
      raise ValueError(
          f'latent_noise.{name} must be bool, got {value!r}.')

  if latent.get('apply_train_value', False):
    raise ValueError('latent_noise.apply_train_value=True is not supported.')
  if latent.get('apply_policy_eval', False):
    raise ValueError('latent_noise.apply_policy_eval=True is not supported.')
  if latent.get('apply_env_acting', False):
    raise ValueError('latent_noise.apply_env_acting=True is not supported.')


def _shadow_direction_stats(adv_t0):
  adv_t0 = f32(adv_t0)
  if adv_t0.shape[0] != 3:
    raise ValueError(
        f'shadow disagreement requires exactly 3 total views, '
        f'got {adv_t0.shape[0]}.')
  adv_t0 = adv_t0.reshape((3, -1))
  mean_abs_adv = jnp.abs(adv_t0).mean(0)
  impact_ref = jnp.maximum(jnp.median(mean_abs_adv), f32(1e-6))
  eps = jnp.maximum(f32(1e-6) * impact_ref, f32(1e-8))
  valid = mean_abs_adv > f32(0.05) * impact_ref
  raw_conflict = 1 - jnp.abs(adv_t0.mean(0)) / (mean_abs_adv + eps)
  conflict = jnp.where(valid, jnp.clip(raw_conflict, 0, 1), 0)
  impact_weight = mean_abs_adv / (mean_abs_adv + impact_ref)
  relstd = jnp.where(
      valid, adv_t0.std(0) / (mean_abs_adv + eps), 0)
  sign_tol = f32(0.05) * impact_ref
  mixed_sign = (
      (adv_t0.min(0) < -sign_tol) &
      (adv_t0.max(0) > sign_tol))
  q75 = jnp.percentile(mean_abs_adv, 75)
  highimpact = mean_abs_adv >= q75
  conflict_high = conflict >= 0.5
  return {
      'mean_abs_adv': mean_abs_adv,
      'impact_ref': impact_ref,
      'eps': eps,
      'valid': valid,
      'conflict': conflict,
      'impact_weight': impact_weight,
      'relstd': relstd,
      'mixed_sign': mixed_sign,
      'highimpact': highimpact,
      'conflict_high': conflict_high,
  }


def _shadow_disagreement_metrics(
    adv_views, ret_views, latent_views, reference_ret, reference_tarval,
    frozen_rscale):
  adv_views = f32(adv_views)
  ret_views = f32(ret_views)
  latent_views = f32(latent_views)
  reference_ret = f32(reference_ret)
  reference_tarval = f32(reference_tarval)
  frozen_rscale = f32(frozen_rscale)
  if adv_views.ndim < 3:
    raise ValueError(
        f'adv_views must have shape (3, ..., horizon), got '
        f'{adv_views.shape}.')
  if ret_views.ndim < 3:
    raise ValueError(
        f'ret_views must have shape (3, ..., horizon), got '
        f'{ret_views.shape}.')
  if latent_views.ndim < 4:
    raise ValueError(
        f'latent_views must have shape (3, ..., horizon + 1, features), '
        f'got {latent_views.shape}.')
  if adv_views.shape[0] != 3:
    raise ValueError(
        f'adv_views must have exactly 3 total views, '
        f'got {adv_views.shape[0]}.')
  if ret_views.shape[0] != 3:
    raise ValueError(
        f'ret_views must have exactly 3 total views, '
        f'got {ret_views.shape[0]}.')
  if latent_views.shape[0] != 3:
    raise ValueError(
        f'latent_views must have exactly 3 total views, '
        f'got {latent_views.shape[0]}.')
  start_shape = adv_views.shape[1:-1]
  horizon = adv_views.shape[-1]
  if ret_views.shape[1:-1] != start_shape or ret_views.shape[-1] != horizon:
    raise ValueError(
        f'ret_views shape {ret_views.shape} is incompatible with '
        f'adv_views shape {adv_views.shape}.')
  if (latent_views.shape[1:-2] != start_shape or
      latent_views.shape[-2] != horizon + 1):
    raise ValueError(
        f'latent_views shape {latent_views.shape} is incompatible with '
        f'adv_views shape {adv_views.shape}.')
  if reference_ret.shape[:-1] != start_shape or reference_ret.shape[-1] != horizon:
    raise ValueError(
        f'reference_ret shape {reference_ret.shape} is incompatible with '
        f'adv_views shape {adv_views.shape}.')
  if (reference_tarval.shape[:-1] != start_shape or
      reference_tarval.shape[-1] != horizon):
    raise ValueError(
        f'reference_tarval shape {reference_tarval.shape} is incompatible '
        f'with adv_views shape {adv_views.shape}.')
  latent_features = int(np.prod(latent_views.shape[-1:]))
  adv_views = adv_views.reshape((3, -1, horizon))
  ret_views = ret_views.reshape((3, -1, horizon))
  latent_views = latent_views.reshape(
      (3, -1, horizon + 1, latent_features))
  reference_ret = reference_ret.reshape((-1, horizon))
  reference_tarval = reference_tarval.reshape((-1, horizon))

  stats = _shadow_direction_stats(adv_views[:, :, 0])
  conflict = stats['conflict']
  impact_weight = stats['impact_weight']
  relstd = stats['relstd']
  eps = stats['eps']

  return_mean_abs = jnp.abs(ret_views).mean(0)
  return_relstd = ret_views.std(0) / (return_mean_abs + eps)
  latent_absdiff = jnp.abs(latent_views[1:] - latent_views[0:1]).mean()
  main_absadv = jnp.abs(reference_ret - reference_tarval) / frozen_rscale

  return {
      'shadow/paired_direction_conflict_weighted_t0_mean':
          (conflict * impact_weight).mean(),
      'shadow/paired_direction_conflict_t0_mean':
          conflict.mean(),
      'shadow/paired_adv_relstd_absdenom_t0_mean':
          relstd.mean(),
      'shadow/paired_adv_relstd_absdenom_t0_p95':
          jnp.percentile(relstd, 95),
      'shadow/paired_mixed_sign_t0_frac':
          f32(stats['mixed_sign']).mean(),
      'shadow/paired_highimpact_conflict_t0_frac':
          f32(stats['highimpact'] & stats['conflict_high']).mean(),
      'shadow/paired_return_relstd_absdenom_mean':
          return_relstd.mean(),
      'shadow/paired_return_relstd_absdenom_p95':
          jnp.percentile(return_relstd, 95),
      'shadow/paired_latent_absdiff_mean':
          latent_absdiff,
      'shadow/main_abs_normalized_advantage_t0_mean':
          main_absadv[:, 0].mean(),
      'shadow/main_abs_normalized_advantage_mean':
          main_absadv.mean(),
      'shadow/main_abs_normalized_advantage_p95':
          jnp.percentile(main_absadv, 95),
  }




def _teacher_gate_mode_code(mode):
  return {
      'disabled': 0,
      'none': 0,
      'baseline': 1,
      'signal': 2,
      'shuffle': 3,
      'uniform': 4,
  }[str(mode).lower()]


def _teacher_gate_alpha_from_conflict(
    conflict_weighted, beta=0.5, alpha_min=0.25, eps=1e-6):
  conflict_weighted = f32(conflict_weighted)
  beta = f32(beta)
  alpha_min = f32(alpha_min)
  eps = f32(eps)
  z = (conflict_weighted - conflict_weighted.mean()) / (
      conflict_weighted.std() + eps)
  alpha = jnp.exp(-beta * jax.nn.relu(z))
  alpha = jnp.clip(alpha, alpha_min, f32(1.0))
  return sg(alpha)


def _apply_teacher_gate_to_policy_loss(policy_loss, policy_gate):
  if policy_gate is None:
    return policy_loss
  gate = sg(f32(policy_gate))
  if gate.ndim == policy_loss.ndim - 1:
    gate = gate[..., None]
  gate = jnp.broadcast_to(gate, policy_loss.shape)
  return policy_loss * gate

def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    policy_gate=None,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  policy_loss = _apply_teacher_gate_to_policy_loss(policy_loss, policy_gate)
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
