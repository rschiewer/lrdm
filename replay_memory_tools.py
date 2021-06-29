import warnings
import os
import gym
import numpy

import numpy as np
import tensorflow
from matplotlib import pyplot as plt, animation
import tensorflow as tf


def detect_trajectories(mem):
    # todo: this is broken, fix it!
    # todo: last state seems to be missing
    end_transitions = np.nonzero(mem['done'])[0]
    # start states are the next state after every terminal state
    start_transitions = end_transitions + 1
    # the last terminal state is not followed by a start state, so use it for first start state
    start_transitions = np.roll(start_transitions, 1)
    start_transitions[0] = 0
    # (start_transition_idx, end_transition_idx, number_transitions)
    traj_info = np.stack([start_transitions, end_transitions, end_transitions - start_transitions], axis=1)
    return traj_info


def detect_trajectories_2(mem):
    start_states = []
    end_states = []
    lengths = []
    current_start_state = None
    for i, transition in enumerate(mem):
        if current_start_state is None:
            start_states.append(i)
            current_start_state = i
        if transition['done']:
            end_states.append(i)
            lengths.append(i - current_start_state)
            current_start_state = None

    if len(start_states) == len(end_states) + 1:
        start_states.pop(-1)  # incomplete last trajectory
    return np.stack([start_states, end_states, lengths], axis=1)


def mem_sanity_check(mem):
    traj_info = detect_trajectories(mem)
    for s, e, l in traj_info:
        #assert mem[s]['done'] == False
        assert mem[e]['done'] == True
        assert np.all(mem[s:e+1]['env'] == mem[s]['env'])
    indices, occurrences = np.unique(mem['env'], return_counts=True)
    print(f'Different environment indices: {indices}, numer of occurrences: {occurrences}')


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
        breakpoint += (tl + 1)
        if breakpoint > desired_length:
            breakpoint -= (tl + 1)
            break
    return mem[0:breakpoint]


def extract_subtrajectories(mem, num_trajectories, traj_length, warn=True, sample_random=True, pad_short_trajectories=False):
    traj_info = detect_trajectories(mem)

    if pad_short_trajectories:
        candidates = np.nonzero(traj_info[:, 2] > 1)[0]
    else:
        candidates = np.nonzero(traj_info[:, 2] >= traj_length)[0]

    # print('Found {} candidate trajectories out of {} total'.format(len(candidates), len(traj_info)))

    if len(candidates) == 0:
        raise ValueError('No trajectories of length {} in memory found'.format(traj_length))

    if len(candidates) < num_trajectories and warn:
        warnings.warn((f'The number of suitable trajectories in memory is {len(traj_info)}, but the requested number '
                       f'of subtrajectories is {num_trajectories}'))

    cand_iter = iter(candidates)
    subtrajectories = np.zeros(shape=(num_trajectories, traj_length), dtype=mem.dtype)
    #subtrajectories['done'] = True  # in case of padded subtrajectories, all padded transitions will be terminal
    n_envs_collected = [0, 0, 0]
    for i_collected in range(num_trajectories):
        i_traj = np.random.choice(candidates) if sample_random else next(cand_iter)
        i_ts, i_te, n_tl = traj_info[i_traj]
        # traj_length + 1 in order to not miss last transition
        if sample_random:
            if pad_short_trajectories:
                i_sub_start = np.random.randint(i_ts, i_te + 1)
                i_sub_end = min(i_te + 1, i_sub_start + traj_length)
            else:
                i_sub_start = np.random.randint(i_ts, i_te - traj_length + 1)
                i_sub_end = i_sub_start + traj_length
        else:
            i_sub_start = 0
            i_sub_end = traj_length

        subtrajectories[i_collected, :i_sub_end - i_sub_start] = mem[i_sub_start: i_sub_end]
        subtrajectories[i_collected, :]['env'] = mem[i_sub_start]['env']
        n_envs_collected[mem[i_sub_start]['env']] += 1
        assert(np.all(mem[i_sub_start: i_sub_end]['env'] == mem[i_sub_start]['env']))


    for st in subtrajectories:
        assert np.sum(st['done']) <= 1

    return subtrajectories


def extract_subtrajectories_unbiased(mem, num_trajectories, traj_length, rand_seed=None):
    if rand_seed:
        tf.random.set_seed(rand_seed)
        np.random.seed(rand_seed)
    start_indices = np.random.default_rng(rand_seed).integers(0, len(mem), num_trajectories)
    subtrajectories = np.zeros(shape=(num_trajectories, traj_length), dtype=mem.dtype)
    for i_subtraj, i_start in enumerate(start_indices):
        terminals = np.nonzero(mem[i_start: i_start + traj_length]['done'])[0]
        if len(terminals) > 0:
            i_end = i_start + terminals[0] + 1
        else:
            i_end = i_start + traj_length
        if i_end > len(mem):
            i_end = len(mem)
        subtrajectories[i_subtraj, 0: i_end - i_start] = mem[i_start: i_end]
        subtrajectories[i_subtraj]['env'] = mem[i_start]['env']
        assert np.all(mem[i_start: i_end]['env'] == mem[i_start]['env'])
    return subtrajectories



def cumulative_episode_rewards(mem):
    traj_info = detect_trajectories(mem)
    rewards = []
    for es, ee, el in traj_info:
        rewards.append(mem[es:ee+1]['r'].sum())
    return rewards


def build_trajectory_from_position_images(traj, actions, condensed_mem, max_diff):
    # find most likely start sample
    start_obs = traj[0]
    i_current = find_closest_match_obs(start_obs[np.newaxis, np.newaxis, ...], condensed_mem, max_diff)[0, 0]
    i_env = condensed_mem[i_current]['env']
    condensed_env_mem = condensed_mem[condensed_mem['env'] == i_env]

    # find new index of same observation in condensed memory
    i_current = find_closest_match_obs(start_obs[np.newaxis, np.newaxis, ...], condensed_env_mem, max_diff)[0, 0]
    pos_current = condensed_env_mem[i_current]['pos']

    # start to build reconstruct trajectory
    observations = []
    traj_finished = False
    for a in actions:
        if a == 0:
            new_pos = pos_current + [0, 1]
        elif a == 1:
            new_pos = pos_current + [1, 0]
        elif a == 2:
            new_pos = pos_current + [0, -1]
        elif a == 3:
            new_pos = pos_current + [-1, 0]
        else:
            raise ValueError(f'Unknown action: {a}')
        i_new = find_closest_match_pos(new_pos[tf.newaxis, tf.newaxis, ...], condensed_env_mem)[0, 0]
        # if no new observation matching the new position could be found, assume the new position is invalid and the
        # agent did not move
        if i_new != -1:
            pos_current = new_pos
            i_current = i_new
            assert condensed_env_mem[i_current]['env'] == i_env

        if traj_finished:
            add_obs = np.zeros_like(condensed_env_mem[i_current]['s_'])
        else:
            add_obs = condensed_env_mem[i_current]['s_']

        if condensed_env_mem[i_current]['done']:
            traj_finished = True

        observations.append(add_obs)

    return i_env, np.array(observations)




def blockworld_position_images(mem):
    n_envs = mem['env'].max() + 1
    gallery = [[] for _ in range(n_envs)]
    env_sizes = []

    # find dimensions of environments
    for i_env in range(n_envs):
        env_samples = mem[mem['env'] == i_env]
        min_x = min((int(pos[0]) for pos in env_samples['pos']))
        min_y = min((int(pos[1]) for pos in env_samples['pos']))
        max_x = max((int(pos[0]) for pos in env_samples['pos']))
        max_y = max((int(pos[1]) for pos in env_samples['pos']))
        env_sizes.append((min_x, max_x + 1, min_y, max_y + 1))
        assert min_x == 0
        assert min_y == 0

    # prepare grid to hold observations for every position
    gallery_size = 0
    for size, cur_env_gallery in zip(env_sizes, gallery):
        for x_coord in range(size[1]):
            cur_env_gallery.append([])
            for y_coord in range(size[3]):
                cur_env_gallery[x_coord].append(None)
                gallery_size += 1

    added_places = 0
    for sample in mem:
        env_idx = sample['env']
        x = int(sample['pos'][0])
        y = int(sample['pos'][1])
        if gallery[env_idx][x][y] is None:
            gallery[env_idx][x][y] = sample['s_']
            added_places += 1

        if added_places == gallery_size:
            break

    return gallery, n_envs, env_sizes


def _run_env(env, test_setting, n_samples, idx):
    print(f'Starting environment {idx + 1}')
    memory = []

    i_episode = 0
    while True:
        last_observation = env.reset()
        t = 0
        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if 'player_pos' in info.keys():
                player_pos = info.pop('player_pos', None)
            elif hasattr(env, 'agent_pos') and env.agent_pos is not None:
                player_pos = env.agent_pos
            else:
                player_pos = (-42, -42)

            if test_setting == 'vizdoom':  # hacky reward normalization for vizdoom
                reward /= 100

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


def gen_data(envs, env_info, samples_per_env, file_paths=None):
    obs_shape = env_info['obs_shape']
    obs_dtype = env_info['obs_dtype']
    test_setting = env_info['test_setting']
    for env in envs:
        assert type(env.action_space) is gym.spaces.Discrete, f'Environment {env} doesn\'t have discrete action space'

    memories = []

    # Collect samples from environment
    for idx, env in enumerate(envs):
        memories.append(_run_env(env, test_setting, samples_per_env, idx))

    # Make numpy arrays out of data
    arr_dtype = np.dtype([('s', obs_dtype, obs_shape),
                          ('a', np.int32, ()),
                          ('r', np.float32, ()),
                          ('s_', obs_dtype, obs_shape),
                          ('done', np.int8, ()),
                          ('pos', np.float32, (2,)),
                          ('env', np.int8, ())])

    for mem in memories:
        np.array(mem, dtype=arr_dtype)
    memories = [np.array(mem, dtype=arr_dtype) for mem in memories]

    if file_paths:
        for path, mem in zip(file_paths, memories):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f'Memory size: {mem.nbytes/1024/1024}')
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
        sub_mem = extract_sub_memory(mem, n_samples)
        mix_memory.append(sub_mem)
        assert sub_mem[-1]['done'] == True

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


def trajectory_video(obs, titles, max_len=np.iinfo(np.int32).max, overall_title=None, max_cols=4, interval=50):
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
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2.1 + 1, n_rows * 2.1 + 1))
    if overall_title:
        fig.suptitle(overall_title)
    to_plot = []

    if n_cols == 1:
        axes = np.array(axes)

    for ax, title in zip(axes.flatten(), titles):
        ax.set_title(title)
    for i in range(min(traj_len, max_len)):
        title = plt.text(0.1, 0.1, i, ha="center", va="center", transform=fig.transFigure, fontsize="large")
        tmp_artists = [title]
        for traj, ax in zip(obs, axes.flatten()):
            ax.set_xticks([])
            ax.set_yticks([])
            tmp_artists.append(ax.imshow(traj[i], animated=True))
        to_plot.append(tmp_artists)

    anim = animation.ArtistAnimation(fig, to_plot, interval=interval, blit=True, repeat_delay=1000)
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


def condense_places_in_mem(mem):
    present_locations = set()
    i_result = []
    for i, sample in enumerate(mem):
        pos_in_env = (sample['env'], *sample['pos'])
        if pos_in_env not in present_locations:
            present_locations.add(pos_in_env)
            i_result.append(i)
    condensed = mem[i_result]
    sorted = np.sort(condensed, order=['env', 'pos'])
    return sorted


def find_closest_match_obs(batch, samples_mem, max_diff=0.01):
    batch_size = tf.shape(batch)[0]
    timesteps = tf.shape(batch)[1]
    num_samples = tf.shape(samples_mem['s_'])[0]
    result_batch = tf.TensorArray(tf.int32, size=batch_size, dynamic_size=False)
    for i_b in range(batch_size):
        result_trajectory = tf.TensorArray(tf.int32, size=timesteps, dynamic_size=False)
        for i_t in range(timesteps):
            obs = batch[i_b, i_t]
            obs_repeated = tf.repeat(obs[tf.newaxis, ...], num_samples, axis=0)
            difference = tf.math.abs(tf.cast(obs_repeated, tf.float32) - tf.cast(samples_mem['s_'], tf.float32))
            per_img_diff = tf.reduce_mean(difference, axis=[1, 2, 3]) / 255.0
            #i_env_0_samples = len(samples_mem[samples_mem['env'] == 0])
            #i_env_1_samples = i_env_0_samples + len(samples_mem[samples_mem['env'] == 1])
            #i_env_2_samples = i_env_1_samples + len(samples_mem[samples_mem['env'] == 2])
            #plt.plot(range(0, i_env_0_samples), per_img_diff[0: i_env_0_samples], label='env 0')
            #plt.plot(range(i_env_0_samples, i_env_1_samples), per_img_diff[i_env_0_samples: i_env_1_samples], label='env 1')
            #plt.plot(range(i_env_1_samples, i_env_2_samples), per_img_diff[i_env_1_samples: i_env_2_samples], label='env 2')
            #plt.legend()
            #plt.show()
            best_match = tf.argmin(per_img_diff, axis=0)
            if per_img_diff[best_match] > max_diff:
                best_match = -1
            result_trajectory = result_trajectory.write(i_t, tf.cast(best_match, dtype=tf.int32))
        result_batch = result_batch.write(i_b, result_trajectory.stack())
    return result_batch.stack()


def find_closest_match_pos(batch, samples_mem, max_diff=0.001):
    batch_size = tf.shape(batch)[0]
    timesteps = tf.shape(batch)[1]
    num_samples = tf.shape(samples_mem['s_'])[0]
    result_batch = tf.TensorArray(tf.int32, size=batch_size, dynamic_size=False)
    for i_b in range(batch_size):
        result_trajectory = tf.TensorArray(tf.int32, size=timesteps, dynamic_size=False)
        for i_t in range(timesteps):
            pos = batch[i_b, i_t]
            best_match = -1
            for i_sample, sample_pos in enumerate(samples_mem['pos']):
                if tf.reduce_mean(tf.math.abs(pos - sample_pos)) < max_diff:
                    best_match = i_sample
                    break
            result_trajectory = result_trajectory.write(i_t, tf.cast(best_match, dtype=tf.int32))
        result_batch = result_batch.write(i_b, result_trajectory.stack())
    return result_batch.stack()
