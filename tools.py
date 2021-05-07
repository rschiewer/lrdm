import pickle
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, List

import blockworld
import gym_minigrid
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from yamldataclassconfig import YamlDataClassConfig

from keras_vq_vae import VectorQuantizerEMAKeras
from predictors import RecurrentPredictor
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from replay_memory_tools import *

from gym import spaces
from gym import ObservationWrapper
import cv2

from replay_memory_tools import condense_places_in_mem, find_closest_match_obs


class FixedSizePixelObs(ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(self, env, obs_size):
        super(FixedSizePixelObs, self).__init__(env)
        self.obs_size = obs_size

        wrapped_observation_space = env.observation_space

        if not isinstance(wrapped_observation_space, spaces.Box):
            raise ValueError("Unsupported observation space structure.")

        # Extend observation space with pixels.
        self.env.reset()
        pixels = self.env.render(mode='rgb_array')
        self.env.close()

        if np.issubdtype(pixels.dtype, np.iextract_subtrajectories_2nteger):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float('inf'), float('inf'))
        else:
            raise TypeError(pixels.dtype)

        new_shape = (64, 64, 3)
        self.observation_space = spaces.Box(shape=new_shape, low=low, high=high, dtype=pixels.dtype)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def render(self, mode='human', **kwargs):
        self.env.render(mode, **kwargs)

    def observation(self, observation):
        obs_img = self.env.render(mode='rgb_array')
        obs_img = obs_img.astype(np.float32) / 255
        obs_resized = cv2.resize(obs_img, self.obs_size, interpolation=cv2.INTER_AREA)
        obs_resized = np.round(obs_resized * 255)
        obs_resized = obs_resized.astype(np.uint8)

        return obs_resized


def gen_environments(test_setting):
    b = blockworld.Blockworld
    if test_setting == 'gridworld_3_rooms':
        env_names = ['Gridworld-partial-room-v0', 'Gridworld-partial-room-v1', 'Gridworld-partial-room-v2']
        environments = [gym.make(env_name) for env_name in env_names]
        obs_shape = environments[0].observation_space.shape
        obs_dtype = environments[0].observation_space.dtype
        n_actions = environments[0].action_space.n
        act_dtype = environments[0].action_space.dtype
    elif test_setting == 'gridworld_doors_teleporters':
        env_names = ['Gridworld-partial-room-v5']
        environments = [gym.make(env_name) for env_name in env_names]
        obs_shape = environments[0].observation_space.shape
        obs_dtype = environments[0].observation_space.dtype
        n_actions = environments[0].action_space.n
        act_dtype = environments[0].action_space.dtype
    elif test_setting == 'gridworld_large_room':
        env_names = ['Gridworld-partial-room-v7']
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
    elif test_setting == 'gym_classics':
        #env_names = ['CartPole-v0', 'LunarLander-v2', 'BipedalWalker-v3']
        env_names = ['CartPole-v0', 'CartPole-v0', 'CartPole-v0']
        environments = [FixedSizePixelObs(gym.make(env_name), (64, 64)) for env_name in env_names]
        #environments = [gym.make(env_name) for env_name in env_names]
        obs_shape = environments[0].observation_space.shape
        obs_dtype = environments[0].observation_space.dtype
        n_actions = environments[0].action_space.n
        act_dtype = environments[0].action_space.dtype
    elif test_setting == 'LunarLander':
        env_names = ['LunarLander-v2']
        environments = [FixedSizePixelObs(gym.make(env_name), (64, 64)) for env_name in env_names]
        # environments = [gym.make(env_name) for env_name in env_names]
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


def vq_vae_net(obs_shape, n_embeddings, d_embeddings, train_data_var, commitment_cost, frame_stack=1, summary=False,
               tf_eager_mode=False):
    assert frame_stack == 1, 'No frame stacking supported currently'
    grayscale_input = obs_shape[-1] == 1
    vae = VectorQuantizerEMAKeras(train_data_var, commitment_cost=commitment_cost, num_embeddings=n_embeddings,
                                  embedding_dim=d_embeddings, grayscale_input=grayscale_input)
    vae.compile(optimizer=tf.optimizers.Adam())

    if not tf_eager_mode:
        vae.build((None, *obs_shape))

    if summary:
        vae.summary()

    return vae


def predictor_net(n_actions, obs_shape, n_envs, vae, det_filters, prob_filters, decider_lw, n_models, tensorboard_log,
                  summary=False, tf_eager_mode=False):
    vae_index_matrix_shape = vae.compute_latent_shape(obs_shape)
    all_predictor = RecurrentPredictor(vae_index_matrix_shape, n_actions,
                                       vae,
                                       open_loop_rollout_training=True,
                                       det_filters=det_filters,
                                       prob_filters=prob_filters,
                                       decider_filters=decider_lw,
                                       n_models=n_models,
                                       n_tasks=n_envs,
                                       debug_log=tensorboard_log)
    all_predictor.compile(optimizer=tf.optimizers.Adam())

    if not tf_eager_mode:
        net_s_obs = tf.TensorShape((None, None, *vae.compute_latent_shape(obs_shape)))
        net_s_act = tf.TensorShape((None, None, 1))
        all_predictor.build([net_s_obs, net_s_act])

    if summary:
        all_predictor.summary()

    return all_predictor


def prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps):
    actions = trajectories['a'].astype(np.int32)
    rewards = trajectories['r'].astype(np.float32)
    terminals = trajectories['done'].astype(np.float32)
    env_idx = trajectories['env']

    batch_size = 32

    obs_datset = (tf.data.Dataset.from_tensor_slices(trajectories['s'])
                  .map(cast_and_normalize_images)
                  .batch(batch_size, drop_remainder=False)
                  .prefetch(-1))
    next_obs_datset = (tf.data.Dataset.from_tensor_slices(trajectories['s_'])
                       .map(cast_and_normalize_images)
                       .batch(batch_size, drop_remainder=False)
                       .prefetch(-1))

    encoded_obs = tf.cast(vae.encode_to_indices(obs_datset), tf.float32)
    encoded_next_obs = tf.cast(vae.encode_to_indices(next_obs_datset), tf.float32)

    encoded_obs = encoded_obs[:, 0:n_warmup_steps]
    encoded_next_obs = encoded_next_obs[:, :n_steps]
    action_inputs = actions[:, 0:n_steps, np.newaxis]
    reward_inputs = rewards[:, 0:n_steps, np.newaxis]
    terminals_inputs = terminals[:, 0:n_steps, np.newaxis]

    return encoded_obs, encoded_next_obs, reward_inputs, action_inputs, terminals_inputs, env_idx


def load_vae_weights(vae, weights_path, train_stats_path=None, plot_training=False, test_memory=None):
    vae.load_weights(weights_path).expect_partial()
    #vae.compile(optimizer=tf.optimizers.Adam())

    if plot_training:
        with open(train_stats_path, 'rb') as handle:
            history = pickle.load(handle)
        for stat_name, stat_val in history.items():
            plt.plot(stat_val, label=stat_name)

        plt.title('VAE train stats')
        plt.legend()
        plt.show()

    if test_memory is not None:
        trajs = None
        while trajs is None:
            try:
                trajs = extract_subtrajectories(test_memory, 3, 100, False)
            except ValueError:
                trajs = None

        if vae.frame_stack == 1:
            obs = cast_and_normalize_images(trajs['s'])
            reconstructed = vae.predict(obs)
            #flattened_obs = np.reshape(obs, (-1, *np.shape(obs)[-3:]))
            #flattened_reconstructed = vae.predict(flattened_obs)
            #reconstructed = np.reshape(flattened_reconstructed, np.shape(obs))
        else:
            stacked_obs = stack_observations(trajs, vae.frame_stack)
            stacked_obs = cast_and_normalize_images(stacked_obs)
            reconstructed = np.clip(vae.predict(stacked_obs), 0, 1)
            reconstructed = unstack_observations(reconstructed, vae.frame_stack)

        reconstructed = cast_and_unnormalize_images(reconstructed)
        ground_truth = trajs['s']

        all_videos = []
        all_titles = []
        for original, rec in zip(ground_truth, reconstructed):
            all_videos.extend([original, rec])
            all_titles.extend(['true', 'predicted'])

        #anim = trajectory_video([trajs['s'] / 255, reconstructed], ['true', 'reconstructed'])
        anim = trajectory_video(all_videos, all_titles, max_cols=2)
        plt.show()


def load_predictor_weights(complete_predictors_list, env_names, plots=False):
    graph_names = ['sp_env_0', 'sp_env_0+1', 'sp_env_0+1+2', 'ap_env_0', 'ap_env_0+1', 'ap_env_0+1+2']
    graph_colors = ['r', 'g', 'b', 'y', 'y', 'y']
    styles = ['-', '-', '-', '-.', '--', ':']

    for i, pred in enumerate(complete_predictors_list):
        pred.load_weights(f'predictors/predictor_{i}')

    if plots:
        with open('predictors/train_stats', 'rb') as handle:
            histories = pickle.load(handle)

        n_epochs, n_subtrajectories, n_steps, n_warmup_steps = histories.pop(0)

        fig = plt.figure(figsize=(14, 6))
        plt.suptitle(f'epochs: {n_epochs}, subtrajectories: {n_subtrajectories}, timesteps: {n_steps}, warmup: {n_warmup_steps}')
        plt.tight_layout()

        plt.subplot(131)
        plt.title(f'Prediction loss')
        for graph_name, col, style, hist in zip(graph_names, graph_colors, styles, histories):
            plt.plot(hist['loss'], color=col, linestyle=style, label=graph_name)

        plt.subplot(132)
        plt.title(f'Prediction mean absolute error')
        for graph_name, col, style, hist in zip(graph_names, graph_colors, styles, histories):
            plt.plot(hist['observation_error'], color=col, linestyle=style, label=graph_name)

        plt.subplot(133)
        plt.title(f'Prediction mean absolute error')
        for graph_name, col, style, hist in zip(graph_names, graph_colors, styles, histories):
            plt.plot(hist['reward_error'], color=col, linestyle=style, label=graph_name)

        plt.legend()
        plt.show()


def predictor_allocation_stability(predictor, mem, vae, i_env):
    gallery, n_envs, env_sizes = blockworld_position_images(mem)
    plot_val = np.full((env_sizes[i_env][1], env_sizes[i_env][3]), -1, dtype=np.float32)
    plot_val_var = plot_val.copy()
    action_names_after_rotation = ['right', 'down', 'left', 'up']
    a = np.full((1, 1, 1), 1)
    for x in range(env_sizes[i_env][1]):
        for y in range(env_sizes[i_env][3]):
            obs = gallery[i_env][x][y]
            if obs is None:
                continue

            # do prediction
            obs = cast_and_normalize_images(obs[np.newaxis, np.newaxis, ...])
            encoded_obs = vae.encode_to_indices(obs)
            o_predicted, r_predicted, done_predicted, w_predictors = predictor.predict([encoded_obs, a])
            most_probable = np.argmax(w_predictors)
            if np.size(w_predictors) == 1:
                pred_entropy = 0
            else:
                pred_entropy = - sum([np.log(w + 1E-5) * w if w > 0 else 0 for w in np.squeeze(w_predictors)])

            # store
            plot_val[x, y] = most_probable
            plot_val_var[x, y] = pred_entropy

    fig, (most_probable_pred, uncertainty) = plt.subplots(1, 2)
    plt.suptitle(f'Environment {i_env}, action {action_names_after_rotation[np.squeeze(a)]}')

    # most probable predictor
    im_0 = most_probable_pred.matshow(plot_val, cmap=plt.get_cmap('Accent'))
    most_probable_pred.title.set_text('Chosen predictor')
    most_probable_pred.get_xaxis().set_visible(False)
    most_probable_pred.get_yaxis().set_visible(False)
    divider_0 = make_axes_locatable(most_probable_pred)
    cax_0 = divider_0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_0, cax=cax_0)

    # predictor probability entropy
    im_1 = uncertainty.matshow(plot_val_var, cmap=plt.get_cmap('inferno'))
    uncertainty.title.set_text('Entropy predictor probabilities')
    uncertainty.get_xaxis().set_visible(False)
    uncertainty.get_yaxis().set_visible(False)
    divider_1 = make_axes_locatable(uncertainty)
    cax_1 = divider_1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_1, cax=cax_1)

    plt.show()

    return fig


def detect_env_per_sample(pred, vae, mem, batch_size, traj_len, max_diff, rand_seed):
    n_envs = mem['env'].max() + 1
    condensed_mem = condense_places_in_mem(mem)

    closest_env_indices = []
    env_weights = []
    for i_env in range(n_envs):
        env_samples = condensed_mem[condensed_mem['env'] == i_env]
        _, _, _, predicted_decoded, _, _, _, chosen_pred = generate_test_rollouts(pred, env_samples, vae, traj_len, 1, batch_size, rand_seed)
        closest_sample_indices = find_closest_match_obs(predicted_decoded, condensed_mem, max_diff).numpy()
        indices = np.full((batch_size, traj_len), -1)
        for i_batch in range(batch_size):
            for i_step in range(traj_len):
                i_mem = closest_sample_indices[i_batch, i_step]
                if i_mem != -1:
                    indices[i_batch, i_step] = condensed_mem[i_mem]['env']

        closest_env_indices.append(indices)
        env_weights.append(chosen_pred)

    return np.array(closest_env_indices), np.array(env_weights)


def plot_env_per_sample(pred, vae, mix_memory, n_trajs, n_time_steps, max_diff, rand_seed):
    indices, weights = detect_env_per_sample(pred, vae, mix_memory, n_trajs, n_time_steps, max_diff, rand_seed)
    n_envs = len(indices)
    timestep_labels = range(len(indices[0, 0]))
    bar_width = 0.5
    for i_env in range(n_envs):
        def count_indices_per_step(indices, n_envs):
            time_steps = indices.shape[1]
            indices_per_step = np.zeros((time_steps, n_envs + 1))
            for i_t in range(time_steps):
                cur_step = indices[:, i_t]
                for i_env in range(n_envs):
                    indices_per_step[i_t, i_env] += np.count_nonzero(cur_step == i_env)
                indices_per_step[i_t, -1] += np.count_nonzero(cur_step == -1)
            return indices_per_step

        per_timestep = count_indices_per_step(indices[i_env], n_envs)
        for plot_i_env in range(n_envs):
            bottom = per_timestep[:, 0:plot_i_env].sum(axis=1)
            plt.bar(timestep_labels, per_timestep[:, plot_i_env], bar_width, bottom=bottom, label=plot_i_env)
        bottom = per_timestep[:, 0:-1].sum(axis=1)
        plt.bar(timestep_labels, per_timestep[:, -1], bar_width, bottom=bottom, label='undefined')
        plt.ylim(bottom=0, top=indices.shape[1] + 1)
        plt.legend()
        plt.show()


def check_traj_correctness(pred, vae, mem, n_trajs, n_time_steps, max_diff, rand_seed):
    _, _, _, predicted_decoded, actions, _, _, chosen_pred = generate_test_rollouts(pred, mem, vae, n_time_steps, 1, n_trajs, rand_seed)
    condensed_mem = condense_places_in_mem(mem)
    traj_differences = []
    env_indices = []
    for pred_traj, real_acts in zip(predicted_decoded, actions):
        i_env, recons_real_traj = build_trajectory_from_position_images(pred_traj, real_acts, condensed_mem, max_diff)
        traj_differences.append(np.mean(np.abs(pred_traj - recons_real_traj)))
        env_indices.append(i_env)
    traj_differences = np.array(traj_differences)
    env_indices = np.array(env_indices)

    env_0_traj_differences = traj_differences[env_indices == 0]
    env_1_traj_differences = traj_differences[env_indices == 1]
    env_2_traj_differences = traj_differences[env_indices == 2]

    plt.bar(['env 0', 'env 1', 'env 2'],
            [env_0_traj_differences.mean(), env_1_traj_differences.mean(), env_2_traj_differences.mean()], 0.35,
            yerr=[env_0_traj_differences.std(), env_1_traj_differences.std(), env_2_traj_differences.std()])
    plt.show()

    #debug_visualize_observation_sequence(trajectory)



def calc_loss(pred, vae, mem, batch_size, traj_len, rand_seed):
    _, _, _, predicted_decoded, _, _, _, chosen_pred = generate_test_rollouts(pred, mem, vae, traj_len, 1, batch_size, rand_seed)


def generate_test_rollouts(predictor, mem, vae, n_steps, n_warmup_steps, n_trajectories, rand_seed=None):
    #if not predictor.open_loop_rollout_training:
    #    n_warmup_steps = n_steps

    trajectories = extract_subtrajectories_unbiased(mem, n_trajectories, n_steps, rand_seed)
    encoded_obs, _, _, actions, _, _ = prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps)

    next_obs = trajectories['s_']
    #if not predictor.open_loop_rollout_training:
    #    targets = next_obs[:, :n_warmup_steps]
    #else:
    #    targets = next_obs[:, n_warmup_steps - 1:]
    targets_obs = next_obs
    targets_r = trajectories['r']
    targets_done = trajectories['done']
    encoded_start_obs = encoded_obs[:, :n_warmup_steps]

    #print([env_idx.shape, encoded_start_obs.shape, one_hot_acts.shape])

    # do rollout
    o_rollout, r_rollout, terminals_rollout, w_predictors = predictor([encoded_start_obs, actions])
    chosen_predictor = np.argmax(tf.transpose(w_predictors, [1, 2, 0]), axis=-1)
    decoded_rollout_obs = cast_and_unnormalize_images(vae.decode_from_indices(o_rollout)).numpy()
    rewards = tf.squeeze(r_rollout).numpy()
    terminals = tf.squeeze(terminals_rollout).numpy()

    return targets_obs, targets_r, targets_done, decoded_rollout_obs, actions, rewards, terminals, chosen_predictor


def rollout_videos(targets, o_rollouts, r_rollouts, done_rollouts, chosen_predictor, video_title, store_animation=False):
    max_pred_idx = chosen_predictor.max() + 1e-5
    max_reward = r_rollouts.max() + 1e-5
    all_videos = []
    all_titles = []
    for i, (ground_truth, o_rollout, r_rollout, done_rollout, pred_weight) in enumerate(zip(targets, o_rollouts, r_rollouts, done_rollouts, chosen_predictor)):
        weight_imgs = np.stack([np.full_like(ground_truth[0], w) / max_pred_idx * 255 for w in pred_weight])
        reward_imgs = np.stack([np.full_like(ground_truth[0], r) / max_reward * 255 for r in r_rollout])
        done_imgs = np.stack([np.full_like(ground_truth[0], done) * 255 for done in done_rollout])
        all_videos.extend([np.clip(ground_truth, 0, 255),
                           np.clip(o_rollout, 0, 255),
                           np.clip(reward_imgs, 0, 255),
                           np.clip(done_imgs, 0, 255),
                           np.clip(weight_imgs, 0, 255)])
        all_titles.extend([f'true {i}', f'o_rollout {i}', f'r_rollout {i}', f'done_rollout {i}', f'weight {i}'])
    anim = trajectory_video(all_videos, all_titles, overall_title=video_title, max_cols=5)

    if store_animation:
        writer = animation.writers['ffmpeg'](fps=10, bitrate=1800)
        anim.save('rollout.mp4', writer=writer)


def debug_visualize_observation_sequence(obs, interval=50):
    # clip and add batch axis
    obs = np.clip(obs, 0, 255)[np.newaxis, ...]
    anim = trajectory_video(obs, ['Debug'], max_cols=1, interval=interval)


def debug_visualize_trajectory(trajs):
    targets = trajs[0]['s'][np.newaxis, ...]
    o_rollout_dummy = trajs[0]['s'][np.newaxis, ...]
    weight_dummy = np.array([0 for _ in range(len(targets[0]))])[np.newaxis, ...]
    rewards = trajs[0]['r'][np.newaxis, ...]
    dones = trajs[0]['done'][np.newaxis, ...]
    rollout_videos(targets, o_rollout_dummy, rewards, dones, weight_dummy, 'Debug')


class ValueHistory:

    def __init__(self, s_val, n_timesteps):
        self.s_val = s_val
        self._max_len = n_timesteps
        self._data = []

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, key, value):
        raise NotImplementedError('Setting items directly is forbidden')

    def append(self, item):
        assert np.shape(item) == self.s_val, f'Expected shape of {self.s_val} but got shape {np.shape(item)}'
        self._data.append(item)
        if len(self._data) > self._max_len:
            self._data.pop(0)

    def to_list(self):
        return self._data

    def to_numpy(self):
        return np.array(self._data)

    def clear(self):
        self._data.clear()


class NeptuneEpochCallback(NeptuneCallback):

    def on_batch_end(self, batch, logs=None):
        pass


class MultiYamlDataClassConfig(YamlDataClassConfig):

    def load(self, file_paths: Union[Path, str, List[Path], List[str]] = None, path_is_absolute: bool = False):
        """
        This method loads from YAML file to properties of self instance.
        Why doesn't load when __init__ is to make the following requirements compatible:
        1. Access config as global
        2. Independent on config for development or use config for unit testing when unit testing
        """
        if file_paths is None:
            file_paths = self.FILE_PATH

        if type(file_paths) is Path or type(file_paths) is str:
            file_paths = [file_paths]

        dict_config = {}
        for file_path in file_paths:
            dict_config.update(yaml.full_load(Path(file_path).read_text('UTF-8')))
        self.__dict__.update(self.__class__.schema().load(dict_config).__dict__)


class SideloadConfigsMixin(YamlDataClassConfig):

    def load(self, path: Union[Path, str] = None, path_is_absolute: bool = False):
        super(SideloadConfigsMixin, self).load(path, path_is_absolute)
        for fld in fields(self):
            path = getattr(self, fld.name)
            print(f'{fld.name}: {path}')
        print('================')
        for fld in fields(self):
            if fld.name is not None and fld.name.startswith('config_') and fld.name.endswith('_path'):
                path = getattr(self, fld.name)
                print(f'{fld.name}: {path}')
                super(SideloadConfigsMixin, self).load(path, False)


@dataclass
class ExperimentConfig(MultiYamlDataClassConfig):
    tf_eager_mode: bool = None
    model_summaries: bool = None
    neptune_project_name: str = None

    env_setting: str = None
    env_n_samples_per_env: int = None
    env_mix_memory_fraction: Union[float, List[float]] = None
    env_sample_mem_path_stub: str = None
    env_mix_mem_path_stub: str = None
    env_name_concat: str = None

    vae_n_train_steps: int = None
    vae_n_steps_per_epoch: int = None
    vae_batch_size: int = None
    vae_commitment_cost: float = None
    vae_decay: float = None
    vae_n_embeddings: int = None
    vae_d_embeddings: int = None
    vae_n_hiddens: int = None
    vae_n_residual_hiddens: int = None
    vae_n_residual_layers: int = None
    vae_frame_stack: int = None
    vae_weights_path: str = None
    vae_train_stats_path: str = None

    pred_n_train_steps: int = None
    pred_n_steps_per_epoch: int = None
    pred_n_trajectories: int = None
    pred_n_traj_steps: int = None
    pred_n_warmup_steps: int = None
    pred_pad_trajectories: bool = None
    pred_det_filters: int = None
    pred_prob_filters: int = None
    pred_decider_lw: int = None
    pred_n_models: int = None
    pred_batch_size: int = None
    pred_tb_log: bool = None
    pred_use_env_idx: bool = None
    pred_weights_path: str = None
    pred_train_stats_path: str = None

    ctrl_n_runs: int = None
    ctrl_n_plan_steps: int = None
    ctrl_n_warmup_steps: int = None
    ctrl_n_rollouts: int = None
    ctrl_n_iterations: int = None
    ctrl_top_perc: float = None
    ctrl_gamma: float = None
    ctrl_act_noise: float = None
    ctrl_consecutive_actions: int = None
    ctrl_max_steps: int = None
    ctrl_render: bool = None

    #FILE_PATH: Path = create_file_path_field(Path(__file__).parent / 'config_general.yml')

