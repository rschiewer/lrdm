import warnings

import os
import gym
import numpy
import pickle
import numpy as np
from matplotlib import pyplot as plt, animation
import tensorflow as tf


def detect_trajectories(mem):
    end_states = np.nonzero(mem['done'])[0]
    # start states are the next state after every terminal state
    start_states = end_states + 1
    # the last terminal state is not followed by a start state, so use it for first start state
    start_states = np.roll(start_states, 1)
    start_states[0] = 0
    # (start_state_idx, end_state_idx, length)
    traj_info = np.stack([start_states, end_states, end_states - start_states], axis=1)
    return traj_info


def extract_sub_memory(mem, desired_length):
    """Tries to extract ``desired_length`` samples from ``mem`` but respects the individual trajectories in
    ``mem``. This method will return only complete trajectories, but tries to keep the number of extracted
    samples as close as possible to ``desired_length``.

    Args:
        mem (Iterable): memory containing the samples in collection order
        desired_length (int): desired number of samples the sub_memory should contain
    """
    traj_info = detect_trajectories(mem)
    breakpoint = 0
    for ts, te, tl in traj_info:
        breakpoint += tl
        if breakpoint > desired_length:
            breakpoint -= tl
            break
    return mem[0:breakpoint]


def extract_subtrajectories(mem, num_trajectories, traj_length, warn=True, random=True):
    traj_info = detect_trajectories(mem)
    candidates = np.nonzero(traj_info[:, 2] >= traj_length)[0]

    # print('Found {} candidate trajectories out of {} total'.format(len(candidates), len(traj_info)))

    if len(candidates) == 0:
        raise ValueError('No trajectories of length {} in memory found'.format(traj_length))

    if len(candidates) < num_trajectories and warn:
        warnings.warn(
            'The number of suitable trajectories in memory is smaller than the requested number of subtrajectories'.format(
                traj_length))

    cand_iter = iter(candidates)
    subtrajectories = np.empty(shape=(num_trajectories, traj_length), dtype=mem.dtype)
    for i in range(num_trajectories):
        traj_idx = np.random.choice(candidates) if random else next(cand_iter)
        ts, te, tl = traj_info[traj_idx]
        # traj_length + 1 in order to not miss last transition
        start_idx = np.random.randint(ts, te - traj_length + 1) if random else 0
        subtrajectories[i] = mem[start_idx: start_idx + traj_length]

    for st in subtrajectories:
        assert np.sum(st['done']) <= 1

    #return np.array(subtrajectories)
    return subtrajectories


def _run_env(env, n_samples, idx):
    print(f'Starting environment {idx + 1}')
    memory = []

    i_episode = 0
    while True:
        last_observation = env.reset()
        t = 0
        #for t in range(env._max_episode_steps):
        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            player_pos = info.pop('player_pos', [-42, -42])
            memory.append((last_observation, action, reward, observation, done, player_pos, idx))

            if done:
                print(f'\tEpisode {i_episode + 1} finished after {t + 1} timesteps')
                break
            else:
                last_observation = observation
            t += 1
        i_episode += 1
        if len(memory) >= n_samples:
            break

    env.close()
    print(f'Environment {idx + 1} done, collected {len(memory)} samples')

    return memory


def gen_data(envs, samples_per_env, file_paths=None):
    obs_shape = envs[0].observation_space.shape
    obs_dtype = envs[0].observation_space.dtype
    acts_shape = envs[0].action_space.shape
    acts_dtype = envs[0].action_space.dtype
    assert type(envs[0].action_space) is gym.spaces.Discrete

    memories = []

    # Collect samples from environment
    for idx, env in enumerate(envs):
        memories.append(_run_env(env, samples_per_env, idx))

    # Make numpy arrays out of data
    arr_dtype = np.dtype([('s', obs_dtype, obs_shape),
                          ('a', acts_dtype, acts_shape),
                          ('r', np.float32, ()),
                          ('s_', obs_dtype, obs_shape),
                          ('done', np.int8, ()),
                          ('pos', np.float32, (2,)),
                          ('env', np.int8, ())])

    memories = [np.array(mem, dtype=arr_dtype) for mem in memories]

    if file_paths:
        for path, mem in zip(file_paths, memories):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb+') as handle:
                np.save(handle, mem, allow_pickle=False)

    return memories


def gen_mixed_memory(memories, mem_fractions, file_path=None):
    assert len(memories) == len(mem_fractions), f'Expected a memory fraction for every memory provided, but got ' \
                                                  f'{len(memories)} memories and {len(mem_fractions)} mix percentages'
    assert all([mp <= 1 for mp in mem_fractions]), 'Memory fractions should be less or equal to 1'

    mix_memory = []
    for mem_frac, mem in zip(mem_fractions, memories):
        n_samples = np.ceil(len(mem) * mem_frac).astype(np.int32)
        mix_memory.append(extract_sub_memory(mem, n_samples))

    mix_memory = np.concatenate(mix_memory, axis=0)

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb+') as handle:
            np.save(handle, mix_memory, allow_pickle=False)

    return mix_memory

    #for i_mem, mem in enumerate(memories):
    #    n_samples = np.ceil(len(mem) / (i_mem + 1)).astype(np.int)
    #    mix_mem = [extract_sub_memory(m, n_samples) for m in memories[0:i_mem + 1]]
    #    simulate_agent_memories.append(np.concatenate(mix_mem, axis=0))

    #if file_paths:
    #    for path, mem in zip(file_paths, simulate_agent_memories):
    #        os.makedirs(os.path.dirname(path), exist_ok=True)
    #        with open(path, 'wb+') as handle:
    #            np.save(handle, mem, allow_pickle=False)

    #return simulate_agent_memories


def cast_and_normalize_images(images):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    images = (tf.cast(images, tf.float32) / 255.0) - 0.5
    return images


def cast_and_unnormalize_images(images):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    char_arrays = tf.cast((images + 0.5) * 255, tf.uint8)
    images = tf.math.minimum(tf.math.maximum(char_arrays, 0), 255)
    return images


def line_up_observations(memory):
    n_samples_final = len(memory) + np.count_nonzero(memory['done'])
    obs_mem = []

    for ts, te, tl in detect_trajectories(memory):
        obs_mem.append(memory['s'][ts:te + 1])
        obs_mem.append(memory['s_'][te, np.newaxis])

    return np.concatenate(obs_mem)


def stack_observations(memory, n_stack):
    current_stack = []
    all_observations = []
    for sample in memory:
        current_stack.append(sample['s'])

        if len(current_stack) == n_stack:
            concatenated = np.concatenate(current_stack, axis=-1)
            all_observations.append(concatenated)
            current_stack = []

        if sample['done']:
            current_stack.append(sample['s_'])
            while len(current_stack) < n_stack:
                current_stack.append(np.zeros_like(current_stack[0]))  # has to be at least one element in stack
            concatenated = np.concatenate(current_stack, axis=-1)
            all_observations.append(concatenated)
            current_stack = []

    return np.array(all_observations, dtype=np.float32)


def unstack_observations(obs, n_stack):
    unstacked = []

    for obs_stack in obs:
        o_unstacked = np.split(obs_stack, n_stack, axis=-1)
        unstacked.extend(o_unstacked)

    return np.array(unstacked, dtype=np.float32)


def trajectory_video(obs, titles, max_len=np.iinfo(np.int32).max, overall_title=None, max_cols=4):
    assert np.ndim(obs) >= 4, ('Please provide at least one trajectory with shape (timestep, with, height, channels), ',
                               f'I got a nested structure with shape {np.shape(obs)} instead.')
    if np.ndim(obs) == 4:
        obs = [obs]
    #assert type(obs) is list, 'Please provide a list with one or more trajectories'

    n_cols = min(len(obs), max_cols)
    traj_len = len(obs[0])

    for i, o in enumerate(obs):
        assert np.ndim(o) == 4, ('Need a 4D nested structure (timestep, obs_w, obs_h, obs_c) for every ',
                             f'trajectory, but trajectory {i} has shape {np.shape(o)} instead')

    n_rows = np.ceil(len(obs) / max_cols).astype(np.int)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2.5 + 1, n_rows * 2.5 + 1))
    if overall_title:
        fig.suptitle(overall_title)
    to_plot = []

    if n_cols == 1:
        axes = np.array(axes)

    #if n_cols == 1:
    #    axes.set_title(titles)
    #    for i in range(min(traj_len, max_len)):
    #        to_plot.append([axes.imshow(obs[i], animated=True)])
    #else:
    for ax, title in zip(axes.flatten(), titles):
        ax.set_title(title)
    for i in range(min(traj_len, max_len)):
        title = plt.text(0.1, 0.1, i, ha="center", va="center", transform=fig.transFigure, fontsize="large")
        tmp_artists = [title]
        for traj, ax in zip(obs, axes.flatten()):
            tmp_artists.append(ax.imshow(traj[i], animated=True))
        to_plot.append(tmp_artists)

    anim = animation.ArtistAnimation(fig, to_plot, interval=50, blit=True, repeat_delay=1000)
    plt.tight_layout()
    plt.show()

    return anim


def load_env_samples(file_names):
    if type(file_names) is str:
        file_names = [file_names]
    assert type(file_names) is list, 'Please provide a string path of a list of string paths'

    print('Loading environment samples')
    memories = []
    for name in file_names:
        with open(name, 'rb') as f:
            memories.append(np.load(f, allow_pickle=False))

    #print('Number of memories: {}'.format(memories.shape[0]))
    #print('Number of samples in memories: {}'.format([len(mem) for mem in memories]))
    #print('Number of samples in simulate agent memories: {}'.format([len(mem) for mem in simulate_agent_memories]))
    #print('Number of elements per sample: {}'.format(len(memories[0].dtype)))
    #print('Observation shape: {}'.format(memories[0]['s'].shape[1:]))
    print('Loading environment samples done')

    if len(memories) == 1:
        memories = memories[0]

    return memories


def _load_env_samples(file_names):
    memories = []
    # TODO: rewrite
    for file_name in file_names:
        with open(f'env_samples/{file_name}_format', 'wb+') as handle:
            arr_dtype = pickle.load(handle)

            columns = []
            for name in arr_dtype.names:
                with open(f'env_samples/{file_name}_{i_mem}_{name}', 'wb+') as f:
                    np.save(f, mem[name])