# Reliability-Guided Actor-Side Gating for DreamerV3

Research implementation by **Renbin He**, developed as a two-term
COMP9991/COMP9993 research project at UNSW Sydney.

**Research release branch:** `comp9993/research-release`

This project extends DreamerV3 to study whether imagined actor updates should
be trusted uniformly when visual observations are incomplete or corrupted.

> This branch is based on Danijar Hafner's DreamerV3 implementation at upstream
> commit [`e3f02248693a79dc8b0ebd62c93683888ddaccfe`](https://github.com/danijar/dreamerv3/tree/e3f02248693a79dc8b0ebd62c93683888ddaccfe).
> The original project description and instructions are preserved in
> [README_DREAMERV3_UPSTREAM.md](README_DREAMERV3_UPSTREAM.md).

## Core idea

The main method uses value inconsistency (VI), defined from the difference
between the critic estimate and the corresponding lambda-return, as a
sample-specific reliability signal for imagined rollouts.

The detached actor weight is computed as:

```text
alpha = exp(-beta * ReLU(VI / median(VI) - threshold))
```

and clipped to a configured minimum value. The weight is applied only to the
imagined actor-loss terms. Therefore, the world-model and critic objectives
remain unchanged.

The implementation also supports disagreement-only, VI-only, hybrid, shuffled,
uniform, and ungated control conditions.

## Engineering contributions

- Configurable visual observation corruption: Gaussian noise, temporally
  correlated pink noise, frame blackout, and mutually exclusive mixed
  corruption.
- Deterministic DMC environment seeding across parallel workers.
- Common-action shadow rollouts for disagreement measurement.
- Actor-policy-only latent perturbation at rollout starts or future imagined
  features.
- Detached sample-specific actor weighting using VI and disagreement signals.
- Matched shuffled and uniform controls for testing signal specificity.
- Runtime metrics for corruption, disagreement, VI, gate activation, and actor
  weights.
- Unit and validation tests for corruption, seeding, diagnostics, and gating.

## Evaluation summary

The primary evaluation used DMC Walker Walk with 600,000 environment steps and
five matched seeds per condition.

In the original mixed-corruption evaluation, VI-only improved the mean training
score by approximately 33.44 points over the ungated baseline, with a positive
matched-seed difference for all five seeds. However, clean-setting costs and
inconsistent transfer to DMC Cheetah Run showed that the benefit was dependent
on the task and training stage.

A subsequent blackout ablation evaluated baseline and VI-only at 10%, 30%,
60%, and 90% blackout. The clearest positive late-training pattern occurred at
60%, although the corrected statistical tests remained inconclusive.
Frozen-checkpoint diagnostics also showed that the current gate response was
not proportional to blackout severity, motivating further mechanism analysis.

Accordingly, this repository presents a research prototype and controlled
evaluation framework rather than claiming universal robustness improvement.

## Code map

| Path | Contribution |
|---|---|
| `dreamerv3/agent.py` | VI, disagreement, and hybrid gates; actor-loss weighting |
| `dreamerv3/configs.yaml` | Reproducible corruption, diagnostic, and gate configurations |
| `dreamerv3/main.py` | Observation-wrapper integration and seed metadata |
| `embodied/core/wrappers.py` | Gaussian, pink-noise, blackout, and mixed corruption |
| `embodied/envs/dmc.py` | Deterministic DMC task seeding |
| `embodied/tests/` | Tests for seeding, corruption, latent noise, disagreement, and gating |
| `docs/comp9991/` | Architecture audit and implementation plans |

## Example configuration

The configuration blocks can be composed using the standard DreamerV3 command
line:

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/vi_gate/{timestamp} \
  --configs dmc_vision size12m mixdrop30 tg0_vi_only \
  --task dmc_walker_walk \
  --seed 111 \
  --run.steps 600000
```

The formal experiments used frozen configurations and matched seeds. Training
checkpoints and large experiment logs are intentionally excluded from this
repository.

## Verification

The project-specific tests are located in:

```text
embodied/tests/test_observation_noise.py
embodied/tests/test_dmc_env_seeding.py
embodied/tests/test_shadow_disagreement.py
embodied/tests/test_latent_noise.py
embodied/tests/test_teacher_gate.py
```

They can be run on CPU with:

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  embodied/tests/test_observation_noise.py \
  embodied/tests/test_dmc_env_seeding.py \
  embodied/tests/test_shadow_disagreement.py \
  embodied/tests/test_latent_noise.py \
  embodied/tests/test_teacher_gate.py
```

## Attribution and licence

This work is an academic research extension of the original DreamerV3
implementation. The upstream code and this derivative repository remain under
the included MIT licence.
