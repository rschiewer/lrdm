import gym
import gym_minigrid


def gen_environments(test_setting):
    if test_setting == 'gridworld_3_rooms':
        env_names = ['Gridworld-partial-room-v0', 'Gridworld-partial-room-v1', 'Gridworld-partial-room-v2']
        environments = [gym.make(env_name) for env_name in env_names]
        obs_shape = environments[0].observation_space.shape
        obs_dtype = environments[0].observation_space.dtype
        n_actions = environments[0].action_space.n
        act_dtype = environments[0].action_space.dtype
    elif test_setting == 'atari':
        env_names = ['BoxingNoFrameskip-v0', 'SpaceInvadersNoFrameskip-v0', 'DemonAttackNoFrameskip-v0']
        # envs = [gym.wrappers.GrayScaleObservation(gym.wrappers.ResizeObservation(gym.make(env_name), obs_resize), keep_dim=True) for env_name in env_names]
        environments = [gym.wrappers.AtariPreprocessing(gym.make(env_name), grayscale_newaxis=True) for env_name in env_names]
        obs_shape = environments[0].observation_space.shape
        obs_dtype = environments[0].observation_space.dtype
        n_actions = environments[0].action_space.n
        act_dtype = environments[0].action_space.dtype
    elif test_setting == 'new_gridworld':
        env_names = ['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-LavaCrossingS9N2-v0', 'MiniGrid-ObstructedMaze-1Dl-v0']
        environments = [gym.wrappers.TransformObservation(gym_minigrid.wrappers.RGBImgPartialObsWrapper(gym.make(env_name)),
                                                          lambda obs: obs['image']) for env_name in env_names]
        obs_shape = environments[0].observation_space['image'].shape
        obs_dtype = environments[0].observation_space['image'].dtype
        n_actions = environments[0].action_space.n
        act_dtype = environments[0].action_space.dtype
    else:
        raise ValueError(f'Unknown test setting: {test_setting}')

    env_info = {'obs_shape': obs_shape, 'obs_dtype': obs_dtype, 'n_actions': n_actions, 'act_dtype': act_dtype}
    return env_names, environments, env_info