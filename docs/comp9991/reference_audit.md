# COMP9991 DreamerV3 reference audit

Scope: read-only architecture audit of the pinned Danijar DreamerV3 repository at upstream commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe` (`docs/comp9991/upstream_commit.txt:1`). This document distinguishes facts found in this repository from recommendations for a later implementation pass. No gate, threshold tuning, `imag_loss` change, source-code change, dependency change, config change, or training-behavior change is proposed here.

Reproducibility note: `python -m pip show jax ninjax elements granular` found no installed versions in the current Local audit shell. Repository requirements are shown here, but the resolved Katana environment must be frozen before instrumentation.

| Package | Current Local audit shell | Repository requirement |
|---|---|---|
| JAX | not installed | `jax[cuda12]==0.4.33` (`requirements.txt:11`) |
| Ninjax | not installed | `ninjax>=3.5.1` (`requirements.txt:13`) |
| Elements | not installed | `elements>=3.19.1` (`requirements.txt:7`) |
| Granular | not installed | `granular>=0.20.3` (`requirements.txt:9`) |

`ninjax` is specified with a lower bound rather than an exact version, so the exact resolved Ninjax version must be recorded before instrumentation. The same is true for `elements` and `granular`.

## 1. DMC visual walker-walk configuration and command

Facts found in code:

- The default script is `train`, default task is overridden by configs, and the default logdir is `~/logdir/{timestamp}` (`dreamerv3/configs.yaml:3`, `dreamerv3/configs.yaml:7`, `dreamerv3/configs.yaml:9`).
- Default DMC env settings are `size: [64, 64]`, `repeat: 1`, `proprio: True`, `image: True`, `camera: -1` (`dreamerv3/configs.yaml:36`).
- `dmc_vision` selects `task: dmc_walker_walk`, disables proprio with `env.dmc.proprio: False`, and sets `run.steps: 1.1e6`, `run.train_ratio: 256` (`dreamerv3/configs.yaml:184`-`dreamerv3/configs.yaml:187`).
- README's training-script pattern is `python dreamerv3/main.py ... --configs <config>` (`README.md:68`-`README.md:78`).

Exact minimal command for this repository's DMC visual walker-walk run:

```sh
python dreamerv3/main.py --configs dmc_vision --task dmc_walker_walk
```

Equivalent explicit-logdir form:

```sh
python dreamerv3/main.py --logdir ~/logdir/dreamer/{timestamp} --configs dmc_vision --task dmc_walker_walk
```

`--task dmc_walker_walk` is redundant with `--configs dmc_vision`, but it makes the target explicit for COMP9991 validation.

## 2. Environment construction and wrapper chain

Facts found in code:

- `dreamerv3.main.make_env()` splits `config.task` into `suite, task`; for `dmc_walker_walk`, `suite == "dmc"` and `task == "walker_walk"` (`dreamerv3/main.py:212`-`dreamerv3/main.py:214`).
- The DMC constructor is resolved from `'embodied.envs.dmc:DMC'` (`dreamerv3/main.py:217`-`dreamerv3/main.py:238`).
- Suite kwargs are loaded from `config.env.dmc`, then overrides are applied (`dreamerv3/main.py:239`-`dreamerv3/main.py:240`).
- `make_env()` creates `env = ctor(task, **kwargs)` and immediately calls `wrap_env(env, config)` (`dreamerv3/main.py:245`-`dreamerv3/main.py:246`).
- `embodied.envs.dmc.DMC.__init__()` sets `MUJOCO_GL=egl` if missing, splits string envs into domain/task, uses default camera `0` for walker because only quadruped and rodent have DMC-specific defaults, and calls `suite.load(domain, task)` for walker (`embodied/envs/dmc.py:21`-`embodied/envs/dmc.py:40`).
- DMC wraps the dm_control env as `from_dm.FromDM`, then `embodied.wrappers.ActionRepeat` (`embodied/envs/dmc.py:41`-`embodied/envs/dmc.py:43`).
- `wrap_env()` then adds, in order: `NormalizeAction` for each continuous action, `UnifyDtypes`, `CheckSpaces`, and finally `ClipAction` for each continuous action (`dreamerv3/main.py:249`-`dreamerv3/main.py:258`).
- `FromDM` adds the `reset` action and emits base keys `reward`, `is_first`, `is_last`, `is_terminal` (`embodied/envs/from_dm.py:31`-`embodied/envs/from_dm.py:49`, `embodied/envs/from_dm.py:72`-`embodied/envs/from_dm.py:78`).

Effective wrapper chain for DMC vision:

`dm_control.suite.load("walker", "walk") -> FromDM -> ActionRepeat(repeat=1) -> DMC image/proprio projection -> NormalizeAction -> UnifyDtypes -> CheckSpaces -> ClipAction`.

## 3. Observation-noise parity and insertion point

Facts found in code:

- `Agent.__init__()` builds `enc_space` by excluding only `is_first`, `is_last`, `is_terminal`, and `reward` from observations (`dreamerv3/agent.py:38`-`dreamerv3/agent.py:43`).
- `Agent.loss()` calls `self.enc(enc_carry, obs, reset, training)` before RSSM observe/loss (`dreamerv3/agent.py:163`-`dreamerv3/agent.py:167`).
- `rssm.Encoder.__call__()` asserts image inputs are `jnp.uint8`, concatenates sorted image keys, casts to compute dtype, divides by 255, and subtracts 0.5 (`dreamerv3/rssm.py:226`-`dreamerv3/rssm.py:231`).

Recommendation for a later implementation pass:

- Exact SheepRL parity requires an observation wrapper in `wrap_env()` after `UnifyDtypes` and before `CheckSpaces`, so corrupted `uint8` images enter replay, the encoder, and reconstruction targets (`dreamerv3/main.py:249`-`dreamerv3/main.py:258`; replay writes transitions at `embodied/run/train.py:57`-`embodied/run/train.py:62`; reconstruction targets are built at `dreamerv3/agent.py:178`-`dreamerv3/agent.py:182`).
- Encoder-internal noise after `/ 255 - 0.5` would be a separate encoder-augmentation or denoising ablation. It is not parity-equivalent because replay and reconstruction targets would still contain clean images.
- Gaussian noise and pink-noise algorithms, clipping, dtype conversion back to `uint8`, temporal reset behavior, and RNG seeding must match the pinned SheepRL implementation exactly. Ambiguous mapping: the SheepRL code is not present in this repository, so those algorithms and reset/RNG semantics cannot be inferred from DreamerV3 alone.

## 4. Image dtype at relevant boundaries

Facts found in code:

| Boundary | Dtype / scale | Evidence |
|---|---:|---|
| DMC obs space image | `np.uint8`, shape `(64, 64, 3)` for defaults | `embodied/envs/dmc.py:55`-`embodied/envs/dmc.py:57` |
| DMC rendered image | `physics.render(...)`; obs space declares `uint8` and `CheckSpaces` later validates | `embodied/envs/dmc.py:71`-`embodied/envs/dmc.py:76`, `embodied/core/wrappers.py:251`-`embodied/core/wrappers.py:257` |
| `UnifyDtypes` output | floating obs become `np.float32`; `uint8` stays `np.uint8` | `embodied/core/wrappers.py:223`-`embodied/core/wrappers.py:240` |
| Agent obs-space image detection | image means `dtype == np.uint8` and rank 3 | `dreamerv3/agent.py:21` |
| Encoder image input | asserts `jnp.uint8` | `dreamerv3/rssm.py:226`-`dreamerv3/rssm.py:230` |
| Encoder CNN input | compute dtype, scaled to `[0, 1]`, shifted to `[-0.5, 0.5]` | `dreamerv3/rssm.py:230`-`dreamerv3/rssm.py:231`; compute dtype default `bfloat16` in `dreamerv3/configs.yaml:72`-`dreamerv3/configs.yaml:79` and `embodied/jax/nets.py:13` |
| Decoder image prediction | sigmoid float in `[0, 1]`; MSE output aggregated over image axes | `dreamerv3/rssm.py:346`-`dreamerv3/rssm.py:356` |
| Reconstruction target | original `uint8` obs cast to float and divided by 255 | `dreamerv3/agent.py:178`-`dreamerv3/agent.py:182` |
| Report video output | prediction multiplied by 255, clipped, cast to `uint8` | `dreamerv3/agent.py:290`-`dreamerv3/agent.py:307` |

## 5. RSSM observe and imagine paths

Facts found in code:

- `RSSM.observe()` casts carry/tokens/action and either calls `_observe()` once for `single=True` or scans `_observe()` over time with `nj.scan(..., axis=1)` (`dreamerv3/rssm.py:61`-`dreamerv3/rssm.py:73`).
- `_observe()` masks deter/stoch/action on resets, concatenates action with `DictConcat`, advances deterministic state with `_core()`, concatenates deterministic state with encoder tokens unless `absolute=True`, produces posterior logits with `_logit("obslogit", x)`, and samples posterior stochastic state with `nj.seed()` (`dreamerv3/rssm.py:75`-`dreamerv3/rssm.py:92`).
- `RSSM.loss()` calls `observe()`, computes prior from deterministic features, then KL losses `dyn` and `rep` with optional free nats (`dreamerv3/rssm.py:120`-`dreamerv3/rssm.py:133`).
- `RSSM.imagine()` with `single=True` samples/uses an action, advances `_core()`, computes prior logits with `_prior()`, and samples prior stochastic state with `nj.seed()` (`dreamerv3/rssm.py:94`-`dreamerv3/rssm.py:104`).
- Batched imagination uses `nj.scan()` either with a callable policy or with a provided action sequence (`dreamerv3/rssm.py:105`-`dreamerv3/rssm.py:118`).

## 6. Imagined actions, rewards, continuation, values, returns, advantages

Facts found in code:

- Starts are chosen with `K = min(self.config.imag_last or T, T)` and `H = self.config.imag_length` (`dreamerv3/agent.py:188`-`dreamerv3/agent.py:191`). Defaults are `imag_last: 0`, `imag_length: 15` (`dreamerv3/configs.yaml:104`-`dreamerv3/configs.yaml:105`), so `K=T`.
- Starting latent states come from `self.dyn.starts(dyn_entries, dyn_carry, K)` (`dreamerv3/agent.py:191`; implementation at `dreamerv3/rssm.py:56`-`dreamerv3/rssm.py:60`).
- Imagined actions are sampled by `policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))`; `sample()` calls each distribution's `.sample(nj.seed())` (`dreamerv3/agent.py:18`, `dreamerv3/agent.py:192`).
- `self.dyn.imagine(starts, policyfn, H, training)` returns imagined features and previous actions (`dreamerv3/agent.py:193`).
- The observed starting feature is prepended, then a final action is sampled from the last imagined feature, producing `imgact` with horizon `H+1` (`dreamerv3/agent.py:194`-`dreamerv3/agent.py:201`).
- Imagined rewards are `self.rew(inp, 2).pred()` (`dreamerv3/agent.py:203`-`dreamerv3/agent.py:206`).
- Continuation probabilities are `self.con(inp, 2).prob(1)` (`dreamerv3/agent.py:203`-`dreamerv3/agent.py:207`).
- Policy/value/slow value predictions for imagined features are passed into `imag_loss()` (`dreamerv3/agent.py:207`-`dreamerv3/agent.py:214`).
- In `imag_loss()`, denormalized value and slow value are `value.pred() * vscale + voffset` and `slowvalue.pred() * vscale + voffset` (`dreamerv3/agent.py:397`-`dreamerv3/agent.py:400`).
- Lambda returns are computed by `lambda_return(last, term, rew, tarval, tarval, disc, lam)` (`dreamerv3/agent.py:401`-`dreamerv3/agent.py:405`; `lambda_return()` at `dreamerv3/agent.py:482`-`dreamerv3/agent.py:490`).
- Advantages are `(ret - tarval[:, :-1]) / rscale`; policy loss uses stopped-gradient weights and stopped-gradient normalized advantages (`dreamerv3/agent.py:407`-`dreamerv3/agent.py:415`).

## 7. Exact semantics of key mechanisms

Facts found in code:

| Mechanism | Semantics |
|---|---|
| `symexp_twohot` | `MLPHead.Head.symexp_twohot()` creates logits with trailing `bins`, builds bin centers by taking a linear grid over `[-20, 0]` or mirrored halves, applies `nets.symexp(x) = sign(x) * expm1(abs(x))`, mirrors around zero, and returns `outs.TwoHot(logits, bins)` (`embodied/jax/heads.py:132`-`embodied/jax/heads.py:144`; `embodied/jax/nets.py:59`-`embodied/jax/nets.py:64`). `TwoHot.pred()` softmaxes logits and computes a symmetric weighted average over bins (`embodied/jax/outs.py:273`-`embodied/jax/outs.py:309`). `TwoHot.loss()` projects a float target onto adjacent bins and applies cross entropy (`embodied/jax/outs.py:311`-`embodied/jax/outs.py:330`). |
| Reward/value bin counts | Default reward head and value head both use `output: symexp_twohot` and `bins: 255` (`dreamerv3/configs.yaml:98`, `dreamerv3/configs.yaml:101`). Debug config overrides `.*\.bins: 5` (`dreamerv3/configs.yaml:204`-`dreamerv3/configs.yaml:213`). |
| Return normalization | `self.retnorm = Normalize(**config.retnorm)` (`dreamerv3/agent.py:70`), default `impl: perc`, `rate: 0.01`, `limit: 1.0`, `perclo: 5.0`, `perchi: 95.0`, `debias: False` (`dreamerv3/configs.yaml:111`). In `imag_loss()`, `retnorm(ret, update)` updates stats and returns offset/scale; advantage divides by `rscale`, while metrics report `(ret - roffset) / rscale` (`dreamerv3/agent.py:407`-`dreamerv3/agent.py:410`, `dreamerv3/agent.py:424`-`dreamerv3/agent.py:437`; `embodied/jax/utils.py:39`-`embodied/jax/utils.py:92`). |
| Value normalization | `self.valnorm = Normalize(**config.valnorm)` (`dreamerv3/agent.py:71`), default `impl: none`, `rate: 0.01`, `limit: 1e-8` (`dreamerv3/configs.yaml:112`). With `impl: none`, stats are `(0.0, 1.0)` (`embodied/jax/utils.py:59`-`embodied/jax/utils.py:64`). `imag_loss()` still calls `valnorm(ret, update)` before constructing normalized value targets (`dreamerv3/agent.py:417`-`dreamerv3/agent.py:423`). |
| Slow value model | `self.slowval` wraps a separate `MLPHead` named `slowval`, tracking `self.val` via `embodied.jax.SlowModel` (`dreamerv3/agent.py:65`-`dreamerv3/agent.py:68`). Default update is `rate: 0.02`, `every: 1` (`dreamerv3/configs.yaml:110`). `SlowModel.update()` performs exponential mixing `rate * source + (1 - rate) * dst` every `every` updates (`embodied/jax/utils.py:94`-`embodied/jax/utils.py:119`). `Agent.train()` calls `self.slowval.update()` after optimizer step (`dreamerv3/agent.py:137`-`dreamerv3/agent.py:143`). |
| Replay value loss | Enabled by default with `repval_loss: True`, gradient through replay feature controlled by `repval_grad: True` (`dreamerv3/configs.yaml:115`-`dreamerv3/configs.yaml:116`). `Agent.loss()` uses last `K` replay features, observations, and bootstraps from imagined return at horizon index 0, then calls `repl_loss()` (`dreamerv3/agent.py:218`-`dreamerv3/agent.py:235`). `repl_loss()` computes lambda returns from replay rewards plus imagined bootstrap and applies a value loss plus slow-value regularizer, weighted by `~last` (`dreamerv3/agent.py:449`-`dreamerv3/agent.py:479`). |
| Reward gradients | `reward_grad: True` by default (`dreamerv3/configs.yaml:114`). `Agent.loss()` uses `inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)` for reward loss, so with default `True`, reward prediction gradients are allowed into representation features; with `False`, `jax.lax.stop_gradient` blocks them (`dreamerv3/agent.py:17`, `dreamerv3/agent.py:172`-`dreamerv3/agent.py:173`). |

## 8. Ninjax/JAX random seeds during imagination

Facts found in code:

- The outer JAX agent creates per-call seeds deterministically from `[config.seed, counter]` with NumPy RNG, returning two `uint32` values (`embodied/jax/agent.py:405`-`embodied/jax/agent.py:408`).
- Training data batches receive a seed in `Agent.stream()` based on `n_batches`, before `Agent.train()` pops the seed and calls the jitted train function (`embodied/jax/agent.py:263`-`embodied/jax/agent.py:276`, `embodied/jax/agent.py:326`-`embodied/jax/agent.py:337`).
- `transform.apply()` passes that seed into the `ninjax.pure` function; under shardmap, it can fold in the data-axis index (`embodied/jax/transform.py:62`-`embodied/jax/transform.py:89`).
- During imagination in `Agent.loss()`, random consumption comes from: policy action sampling via `sample(... nj.seed())` (`dreamerv3/agent.py:18`, `dreamerv3/agent.py:192`), RSSM stochastic prior sampling in `RSSM.imagine(single=True)` (`dreamerv3/rssm.py:94`-`dreamerv3/rssm.py:104`), and the final `lastact` policy sample (`dreamerv3/agent.py:197`-`dreamerv3/agent.py:199`).
- During representation learning before imagination, posterior sampling in `_observe()` also consumes `nj.seed()` (`dreamerv3/rssm.py:75`-`dreamerv3/rssm.py:92`).

Recommendation for a later implementation pass:

- To avoid changing the main training RNG stream, do not insert extra `nj.seed()` consumers into `Agent.loss()` before or between existing observe/imagine/action calls.
- Put shadow evaluation in a separate pure/report-style path or explicitly run it after all existing loss terms have been computed, with its own derived seed stream that is not consumed by `imag_loss` or optimizer objectives.
- Shadow evaluations must not update `retnorm`, `valnorm`, `advnorm`, model state, losses, or optimizer objectives. All auxiliary outputs must be stop-gradient and metrics-only.
- An RNG-invariance test is required before running experiments. The test should compare the existing main losses/actions/returns for a fixed seed and fixed replay batch before and after instrumentation.
- Exact Ninjax seed isolation API is ambiguous from this repository because `ninjax` is an external dependency (`requirements.txt:13`) and its implementation is not vendored here.

## 9. K=3 shadow imagined evaluations without changing main RNG or objectives

Facts found in code:

- Main optimization objective is the sum over `losses` scaled by `self.scales` (`dreamerv3/agent.py:237`-`dreamerv3/agent.py:245`).
- `imag_loss()` returns losses, `outs['ret']`, and metrics; it does not expose a hook for auxiliary shadow metrics (`dreamerv3/agent.py:382`-`dreamerv3/agent.py:446`).
- Logger/report metrics are separate from optimizer losses; `Agent.report()` calls `self.loss(..., training=False)` and then constructs report artifacts (`dreamerv3/agent.py:247`-`dreamerv3/agent.py:310`).

Recommendation for a later implementation pass:

- Create K=3 shadow imagined evaluations in a metrics-only path, not by adding terms to `losses` and not by changing `imag_loss()`.
- Use the already-computed main start states and `sg(imgprevact)` transition actions where possible, and run shadow rollouts under an isolated seed path. If implemented inside the same jitted train call, schedule it after existing main-loss computations and ensure outputs are only stop-gradient metrics.
- Auxiliary advantages must use each view's own return and each view's own value baseline, while sharing the main rollout's already-computed return-normalization scale. They must not update `retnorm`, `valnorm`, or `advnorm`.
- Ambiguous mapping: whether SheepRL's K=3 evaluations are supposed to reuse DreamerV3's world-model heads exactly, use detached predicted rewards/values, or run as report-only diagnostics is not specified by the supplied formulas.

## 10. Paired common-action auxiliary views

Facts found in code:

- `RSSM.imagine()` supports two modes: a callable policy or a provided action sequence (`dreamerv3/rssm.py:105`-`dreamerv3/rssm.py:118`).
- When a provided action sequence is passed, each scan step calls `self.imagine(c, a, 1, training, single=True)`, so actions can be shared while RSSM stochastic samples are independently drawn via `nj.seed()` at `dreamerv3/rssm.py:100`.
- `Agent.loss()` already constructs the main imagined action sequence `imgact` after appending `lastact` (`dreamerv3/agent.py:193`-`dreamerv3/agent.py:201`).

Recommendation for paired common-action views:

- Same starting latent state: use the same `starts` object from `self.dyn.starts(...)` (`dreamerv3/agent.py:191`).
- Same detached action sequence: pass `sg(imgprevact)`, the `H` transition actions returned by the main `self.dyn.imagine(starts, policyfn, H, training)` call, into the non-callable `RSSM.imagine()` path (`dreamerv3/agent.py:193`, `dreamerv3/agent.py:199`). Do not pass the full `H+1` `imgact` into an `H`-step RSSM scan.
- Independently sampled RSSM stochastic outcomes: run separate `RSSM.imagine(starts, action_sequence, H, training)` calls under separate isolated seeds, because `RSSM.imagine(single=True)` samples stochastic prior state at `dreamerv3/rssm.py:100`.
- Auxiliary advantages must use each auxiliary view's own lambda return and value baseline, but share the main rollout's already-computed return-normalization scale. They must be stop-gradient, metrics-only values and must not feed into policy/value/replay losses.
- Ambiguous mapping: SheepRL does not specify whether "same detached action sequence" includes the final bootstrap action used only for policy log-prob/value loss alignment in DreamerV3.

## 11. Tensor shapes

Facts found in code:

- Default batch `B=16`, sequence `T=64`, replay context `C=1`, imagination `H=15`, `K=T=64` for default `imag_last: 0` (`dreamerv3/configs.yaml:10`-`dreamerv3/configs.yaml:15`, `dreamerv3/configs.yaml:104`-`dreamerv3/configs.yaml:105`, `dreamerv3/agent.py:188`-`dreamerv3/agent.py:191`).
- Default RSSM dimensions are `deter=8192`, `stoch=32`, `classes=64`, so feature tensor dimension after flattening stoch is `8192 + 32 * 64 = 10240` (`dreamerv3/configs.yaml:89`-`dreamerv3/configs.yaml:91`, `dreamerv3/agent.py:51`-`dreamerv3/agent.py:53`).
- DMC action shape is obtained from dm_control's `action_spec()` through `FromDM.act_space`; the exact walker action dimension is not hard-coded in this repository (`embodied/envs/from_dm.py:42`-`embodied/envs/from_dm.py:49`). Use `A = act_space['action'].shape`.

Shape table:

| Tensor | Shape |
|---|---|
| Replay obs image | `[B, T, 64, 64, 3]` after replay context is removed in `_apply_replay_context()` (`dreamerv3/agent.py:312`-`dreamerv3/agent.py:340`) |
| Posterior features `repfeat['deter']` | `[B, T, 8192]` |
| Posterior features `repfeat['stoch']` | `[B, T, 32, 64]` |
| Posterior logits `repfeat['logit']` | `[B, T, 32, 64]` |
| Starts | `[B*K, ...]` = `[1024, ...]` by reshaping last `K` entries (`dreamerv3/rssm.py:56`-`dreamerv3/rssm.py:60`) |
| Imagined features before prepending start | `[B*K, H, ...]` (`dreamerv3/rssm.py:105`-`dreamerv3/rssm.py:118`) |
| Imagined features after prepending start | `[B*K, H+1, ...]`; asserted in code (`dreamerv3/agent.py:194`-`dreamerv3/agent.py:201`) |
| `feat2tensor(imgfeat)` | `[B*K, H+1, 10240]` (`dreamerv3/agent.py:51`-`dreamerv3/agent.py:53`, `dreamerv3/agent.py:202`) |
| Imagined actions `imgact['action']` | `[B*K, H+1, *A]`; asserted by tree leaves (`dreamerv3/agent.py:197`-`dreamerv3/agent.py:201`) |
| Imagined rewards `rew` | `[B*K, H+1]` (`dreamerv3/agent.py:203`-`dreamerv3/agent.py:206`) |
| Continuations `con` | `[B*K, H+1]` (`dreamerv3/agent.py:203`-`dreamerv3/agent.py:207`) |
| Denormalized values `val`, `slowval`, `tarval` | `[B*K, H+1]` (`dreamerv3/agent.py:397`-`dreamerv3/agent.py:400`) |
| Lambda returns `ret` | `[B*K, H]`; `lambda_return()` drops the final bootstrap step (`dreamerv3/agent.py:482`-`dreamerv3/agent.py:490`) |
| Advantages `adv` / `adv_normed` | `[B*K, H]` (`dreamerv3/agent.py:407`-`dreamerv3/agent.py:410`) |
| Weights `weight` | raw `[B*K, H+1]`; loss uses `weight[:, :-1]` = `[B*K, H]` (`dreamerv3/agent.py:401`-`dreamerv3/agent.py:414`) |
| Policy/value losses before reduction | `[B*K, H]` (`dreamerv3/agent.py:411`-`dreamerv3/agent.py:423`) |
| Added imag losses in `losses` dict | `[B, K]` after `mean(1).reshape((B, K))` (`dreamerv3/agent.py:215`) |
| Replay-value bootstrap `boot` | `[B, K]`, from `imgloss_out['ret'][:, 0].reshape(B, K)` (`dreamerv3/agent.py:221`-`dreamerv3/agent.py:224`) |
| Replay lambda returns | `[B, K-1]` from `repl_loss()` and `lambda_return()` (`dreamerv3/agent.py:449`-`dreamerv3/agent.py:479`) |

## 12. Metric emission locations

Facts found in code:

- `make_logger()` always appends `elements.logger.TerminalOutput(config.logger.filter, 'Agent')` (`dreamerv3/main.py:152`-`dreamerv3/main.py:158`).
- If `jsonl` is in `config.logger.outputs`, it emits `metrics.jsonl` and `scores.jsonl` with score filtering (`dreamerv3/main.py:158`-`dreamerv3/main.py:162`).
- If `tensorboard` is in outputs, it emits `TensorBoardOutput(logdir, config.logger.fps)` (`dreamerv3/main.py:163`-`dreamerv3/main.py:165`).
- If `scope` is in outputs, it emits `ScopeOutput(elements.Path(logdir))` (`dreamerv3/main.py:175`-`dreamerv3/main.py:176`).
- Defaults are `[jsonl, scope]`; TensorBoard is available but not default (`dreamerv3/configs.yaml:22`-`dreamerv3/configs.yaml:26`).
- Training loop adds train metrics, episode stats, replay stats, usage, FPS, timer, then calls `logger.write()` (`embodied/run/train.py:106`-`embodied/run/train.py:114`).
- Train-eval loop does analogous metric emission at `embodied/run/train_eval.py:145`-`embodied/run/train_eval.py:153`.
- README confirms terminal, JSONL, Scope by default and TensorBoard can be enabled (`README.md:87`-`README.md:101`).

## 13. Minimal testing plan

Recommendations for a later implementation pass:

- Static checks: assert no changed files outside intended COMP9991 implementation docs/code; inspect all `nj.seed()` call additions; check no changes to `imag_loss()` signature or objective losses.
- CPU debug run, engineering-only and provisional until tested on Katana: `python dreamerv3/main.py --configs dmc_vision debug --task dmc_walker_walk --jax.platform cpu --run.steps 20 --run.envs 1 --run.train_ratio 1 --run.save_every -1` as a compile/runtime sanity check. This is not expected to learn and cannot be used for scientific comparisons because `debug` changes two-hot bins to 5 (`dreamerv3/configs.yaml:204`-`dreamerv3/configs.yaml:213`).
- GPU engineering smoke, provisional until tested on Katana: `python dreamerv3/main.py --configs dmc_vision size1m --task dmc_walker_walk --run.steps 200 --run.envs 1 --run.train_ratio 8 --run.save_every -1 --jax.prealloc False` on one A100, checking compile, memory, and first train metrics. This uses `dmc_vision size1m` but retains 255 two-hot bins unless explicitly overridden.
- Metric-presence test: verify `metrics.jsonl` includes any new auxiliary metric keys while standard keys such as `train/loss/policy`, `train/loss/value`, `train/adv`, `train/ret`, and `episode/score` remain present.
- RNG-invariance test: with the same seed and a fixed replay batch, compare original vs instrumented main outputs for existing losses/actions/returns before auxiliary metrics are read. This should be a bitwise or tolerance-based test depending on dtype and device.

No real gate and no threshold tuning are proposed.

## 14. Experiment tiers

Facts found in code:

- `dmc_vision` alone inherits the default model scale because it does not apply any `size*` override (`dreamerv3/configs.yaml:85`-`dreamerv3/configs.yaml:118`, `dreamerv3/configs.yaml:184`-`dreamerv3/configs.yaml:187`). The default RSSM has `deter: 8192`, `hidden: 1024`, `stoch: 32`, and `classes: 64` (`dreamerv3/configs.yaml:89`-`dreamerv3/configs.yaml:91`), matching the repository's `size200m` scale entry (`dreamerv3/configs.yaml:145`-`dreamerv3/configs.yaml:148`).
- Scientific reference runs should use `dmc_vision` and the default 255-bin reward/value two-hot heads (`dreamerv3/configs.yaml:98`, `dreamerv3/configs.yaml:101`, `dreamerv3/configs.yaml:184`-`dreamerv3/configs.yaml:187`).
- `size1m` changes model dimensions only, via wildcard overrides for RSSM, depth, and units (`dreamerv3/configs.yaml:120`-`dreamerv3/configs.yaml:123`); it does not override `.*\.bins`.
- The `debug` config changes `.*\.bins` to 5 and is engineering-only (`dreamerv3/configs.yaml:204`-`dreamerv3/configs.yaml:213`).

Recommendations:

- Use `dmc_vision` with 255 bins for scientific comparisons.
- Use `dmc_vision size1m` with 255 bins for GPU engineering smoke when full `dmc_vision` is too expensive.
- Treat all smoke commands in this document as provisional until tested on Katana.
- Do not use `debug` results for scientific comparisons.

## 15. Memory and compute risks for `dmc_vision` on one A100

Facts found in code:

- Default `dmc_vision` alone uses the 200M-scale default model, not the `size1m` override used by `dmc_proprio` (`dreamerv3/configs.yaml:120`-`dreamerv3/configs.yaml:148`, `dreamerv3/configs.yaml:178`-`dreamerv3/configs.yaml:187`).
- Default compute dtype is `bfloat16`, platform `cuda`, preallocation enabled (`dreamerv3/configs.yaml:72`-`dreamerv3/configs.yaml:79`).
- Precompile prints train/report cost and memory analysis if available (`embodied/jax/agent.py:185`-`embodied/jax/agent.py:194`, `embodied/jax/agent.py:481`-`embodied/jax/agent.py:493`).

Risk estimates:

- `feat2tensor(imgfeat)` alone is approximately `B*K*(H+1)*10240*2 bytes = 16*64*16*10240*2 ~= 335 MB` in bfloat16, before head activations, gradients, optimizer state, RSSM logits/stoch tensors, and replay/report buffers.
- RSSM `deter` imagined features alone are about `16*64*16*8192*2 ~= 268 MB`; stochastic one-hot and logits add roughly `67 MB` each at the same horizon and dtype.
- The 200M-scale default plus optimizer state is the main risk, not DMC image storage. An A100 should be plausible, but JAX preallocation and compile-time temporaries can still cause out-of-memory, especially if K=3 shadow rollouts are materialized inside the training step.
- Shadow K=3 views can multiply imagination activations if stored simultaneously. Recommendation: compute metrics with rematerialization/streaming or report-only paths and avoid retaining gradients.

## 16. DreamerV3 vs supplied SheepRL mechanism

Only the following SheepRL mechanism is supplied in the request:

- `inconsistency = abs(lambda_return - baseline)`
- `paired direction conflict = 1 - abs(mean(adv_views)) / mean(abs(adv_views))`

No SheepRL source code is present in this repository (`rg` found no `SheepRL` references). The SheepRL-to-JAX mapping is therefore ambiguous wherever a term is not defined by the supplied formulas.

| Concept | DreamerV3 semantics in this repo | Supplied SheepRL mechanism | Mapping status |
|---|---|---|---|
| Lambda return | `lambda_return(last, term, rew, val, boot, disc, lam)` with `rew[:, 1:]`, terminal/live masks, last masks, and bootstrap (`dreamerv3/agent.py:482`-`dreamerv3/agent.py:490`) | `lambda_return` appears in `abs(lambda_return - baseline)` | Ambiguous: SheepRL horizon indexing, terminal masking, discount, and bootstrap source are not specified. |
| Baseline | In imagination, baseline-like value is `tarval[:, :-1]` for advantages; target value is `slowval` only if `slowtar=True`, default `False` (`dreamerv3/agent.py:397`-`dreamerv3/agent.py:408`; `dreamerv3/configs.yaml:108`) | `baseline` in inconsistency formula | Ambiguous: could map to DreamerV3 `tarval`, `val`, `slowval`, normalized value, or SheepRL-specific baseline. |
| Inconsistency | No equivalent metric exists in this repo. Returns are emitted only as training metrics and `outs['ret']` from `imag_loss()` (`dreamerv3/agent.py:424`-`dreamerv3/agent.py:446`) | `abs(lambda_return - baseline)` | Recommendation only: compute as metrics from detached auxiliary views; do not add gate or objective. |
| Advantage | DreamerV3 `adv = (ret - tarval[:, :-1]) / rscale`, then optional adv normalization; default `advnorm` is `none` (`dreamerv3/agent.py:407`-`dreamerv3/agent.py:410`; `dreamerv3/configs.yaml:113`) | `adv_views` in paired direction conflict | Ambiguous: SheepRL may expect raw, return-normalized, value-normalized, or per-view normalized advantages. |
| Paired direction conflict | No existing paired-view metric. Existing policy loss uses a single imagined trajectory and stopped-gradient normalized advantage (`dreamerv3/agent.py:411`-`dreamerv3/agent.py:415`) | `1 - abs(mean(adv_views)) / mean(abs(adv_views))` | Ambiguous: reduction axes for `mean`, epsilon handling when denominator is zero, and view count K are unspecified. |
| Common actions | `RSSM.imagine()` can accept an action sequence instead of a policy (`dreamerv3/rssm.py:111`-`dreamerv3/rssm.py:114`) | "paired common-action auxiliary views" in request | Plausible mapping: same `starts`, detached action sequence, separate RSSM samples. Ambiguous whether actions come from policy mean, sampled main trajectory, or SheepRL action buffer. |
| Objective interaction | DreamerV3 objective is only `losses` scaled by `self.scales`; `imag_loss()` should remain unchanged (`dreamerv3/agent.py:237`-`dreamerv3/agent.py:245`) | Not specified | Recommendation: metrics-only, no gate, no threshold tuning, no objective change. |
