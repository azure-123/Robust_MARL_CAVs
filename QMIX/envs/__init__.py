from functools import partial
# from .multiagentenv import MultiAgentEEnv
# from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import highway_env
from highway_env.envs import MergeEnvMARL
import gym
from highway_env.envs.common.abstract import AbstractEnv

# def env_fn(env, **kwargs) -> AbstractEnv:
#     # env_args = kwargs.get("env_args", {})
#     return env(**kwargs)

# REGISTRY = {}
# # env = 
# # print(StarCraft2Env)
# # print(Env)
# REGISTRY["highway"] = partial(env_fn, env=AbstractEnv)
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# # REGISTRY["highway"] = partial(env_fn, env=highway_env)

# # if sys.platform == "linux":
# #     os.environ.setdefault("SC2PATH",
# #                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

# env = gym.make('merge-multi-agent-v2')
# env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
