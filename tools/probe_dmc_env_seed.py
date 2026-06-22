import argparse
import pathlib
import sys

import elements
import numpy as np
import ruamel.yaml as yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreamerv3 import main as dreamer_main  # noqa: E402


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', default=['dmc_vision', 'clean'])
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--env-index', type=int, default=0)
  parser.add_argument('--steps', type=int, default=100)
  parser.add_argument(
      '--logdir', default='./scratch/comp9991-dmc-seed-probe')
  args = parser.parse_args(argv)

  steps = max(args.steps, 100)
  config = load_config(args.configs, args.seed, args.logdir)

  same_a = collect_trajectory(config, args.env_index, steps)
  same_b = collect_trajectory(config, args.env_index, steps)
  assert trajectories_equal(same_a, same_b)
  print('SAME_SEED_TRAJECTORY_EXACT_OK')

  different_index = collect_trajectory(config, args.env_index + 1, steps)
  assert not trajectories_equal(same_a, different_index)
  print('DIFFERENT_INDEX_DIVERGENCE_OK')

  different_seed_config = config.update(seed=args.seed + 1)
  different_seed = collect_trajectory(
      different_seed_config, args.env_index, steps)
  assert not trajectories_equal(same_a, different_seed)
  print('DIFFERENT_SEED_DIVERGENCE_OK')

  print('DMC_ENV_SEED_PROBE_OK')


def load_config(names, seed, logdir):
  configs = yaml.YAML(typ='safe').load(
      (dreamer_main.folder / 'configs.yaml').read_text())
  config = elements.Config(configs['defaults'])
  for name in names:
    config = config.update(configs[name])
  return config.update(seed=seed, logdir=logdir)


def collect_trajectory(config, index, steps):
  env = dreamer_main.make_env(config, index)
  try:
    trajectory = []
    obs = env.step(zero_action(env, reset=True))
    trajectory.append(select_fields(obs))
    for _ in range(steps):
      obs = env.step(zero_action(env, reset=False))
      trajectory.append(select_fields(obs))
    return trajectory
  finally:
    env.close()


def zero_action(env, reset):
  action = {}
  for key, space in env.act_space.items():
    if key == 'reset':
      action[key] = np.asarray(reset, dtype=space.dtype)
    else:
      action[key] = np.zeros(space.shape, dtype=space.dtype)
  return action


def select_fields(obs):
  return {
      'image': np.asarray(obs['image']).copy(),
      'reward': np.asarray(obs['reward']).copy(),
      'is_first': np.asarray(obs['is_first']).copy(),
      'is_last': np.asarray(obs['is_last']).copy(),
      'is_terminal': np.asarray(obs['is_terminal']).copy(),
  }


def trajectories_equal(left, right):
  if len(left) != len(right):
    return False
  for lhs, rhs in zip(left, right):
    if lhs.keys() != rhs.keys():
      return False
    for key in lhs:
      if not np.array_equal(lhs[key], rhs[key]):
        return False
  return True


if __name__ == '__main__':
  main()
