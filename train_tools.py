import pickle

import gym
import gym_minigrid
import blockworld
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from keras_vq_vae import VectorQuantizerEMAKeras
from predictors import RecurrentPredictor
from replay_memory_tools import cast_and_normalize_images, extract_subtrajectories, stack_observations, \
    unstack_observations, cast_and_unnormalize_images, trajectory_video, blockworld_position_images


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


def vq_vae_net(obs_shape, n_embeddings, d_embeddings, train_data_var, commitment_cost, frame_stack=1, summary=False):
    assert frame_stack == 1, 'No frame stacking supported currently'
    grayscale_input = obs_shape[-1] == 1
    vae = VectorQuantizerEMAKeras(train_data_var, commitment_cost=commitment_cost, num_embeddings=n_embeddings,
                                  embedding_dim=d_embeddings, grayscale_input=grayscale_input)
    vae.compile(optimizer=tf.optimizers.Adam())

    if summary:
        vae.build((None, *obs_shape))
        vae.summary()

    return vae


def predictor_net(n_actions, obs_shape, vae, det_filters, prob_filters, decider_lw, n_models, tensorboard_log,
                  summary=False):
    vae_index_matrix_shape = vae.compute_latent_shape(obs_shape)
    all_predictor = RecurrentPredictor(vae_index_matrix_shape, n_actions,
                                       vae,
                                       open_loop_rollout_training=True,
                                       det_filters=det_filters,
                                       prob_filters=prob_filters,
                                       decider_lw=decider_lw,
                                       n_models=n_models, debug_log=tensorboard_log)
    all_predictor.compile(optimizer=tf.optimizers.Adam())

    if summary:
        net_s_obs = tf.TensorShape((None, None, *vae.compute_latent_shape(obs_shape)))
        net_s_act = tf.TensorShape((None, None, 1))
        all_predictor.build([net_s_obs, net_s_act])
        all_predictor.summary()

    return all_predictor


def prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps):
    actions = trajectories['a'].astype(np.int32)
    rewards = trajectories['r'].astype(np.float32)
    terminals = trajectories['done'].astype(np.float32)

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

    return encoded_obs, encoded_next_obs, reward_inputs, action_inputs, terminals_inputs


def load_vae_weights(vae, test_memory, file_name, plots=False):
    vae.load_weights('vae_model/' + file_name).expect_partial()
    vae.compile(optimizer=tf.optimizers.Adam())
    with open('vae_model/' + file_name + '_train_stats', 'rb') as handle:
        history = pickle.load(handle)

    if plots:
        for stat_name, stat_val in history.items():
            plt.plot(stat_val, label=stat_name)

        plt.title('VAE train stats')
        plt.legend()
        plt.show()

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
    a = np.full((1, 1, 1), 0)
    for x in range(env_sizes[i_env][1]):
        for y in range(env_sizes[i_env][3]):
            obs = gallery[i_env][x][y]
            if obs is None:
                continue

            # do prediction
            obs = cast_and_normalize_images(obs[np.newaxis, np.newaxis, ...])
            encoded_obs = vae.encode_to_indices(obs)
            o_predicted, r_predicted, w_predictors = predictor.predict([encoded_obs, a])
            most_probable = np.argmax(w_predictors)
            if np.size(w_predictors) == 1:
                pred_entropy = 0
            else:
                pred_entropy = - sum([np.log(w) * w for w in np.squeeze(w_predictors)])

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


def generate_test_rollouts(predictor, mem, vae, n_steps, n_warmup_steps, n_trajectories):
    #if not predictor.open_loop_rollout_training:
    #    n_warmup_steps = n_steps

    trajectories = extract_subtrajectories(mem, n_trajectories, n_steps, warn=False, pad_short_trajectories=True)
    encoded_obs, _, _, actions, _ = prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps)

    next_obs = trajectories['s_']
    #if not predictor.open_loop_rollout_training:
    #    targets = next_obs[:, :n_warmup_steps]
    #else:
    #    targets = next_obs[:, n_warmup_steps - 1:]
    targets = next_obs
    encoded_start_obs = encoded_obs[:, :n_warmup_steps]

    #print([env_idx.shape, encoded_start_obs.shape, one_hot_acts.shape])

    # do rollout
    o_rollout, r_rollout, terminals_rollout, w_predictors = predictor([encoded_start_obs, actions])
    chosen_predictor = np.argmax(tf.transpose(w_predictors, [1, 2, 0]), axis=-1)
    decoded_rollout_obs = cast_and_unnormalize_images(vae.decode_from_indices(o_rollout)).numpy()
    rewards = r_rollout.numpy()
    terminals = terminals_rollout.numpy()

    return targets, decoded_rollout_obs, rewards, terminals, chosen_predictor


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


def _debug_visualize_trajectory(trajs):
    targets = trajs[0]['s'][np.newaxis, ...]
    o_rollout_dummy = trajs[0]['s'][np.newaxis, ...]
    weight_dummy = np.array([0 for _ in range(len(targets[0]))])[np.newaxis, ...]
    rewards = trajs[0]['r'][np.newaxis, ...]
    rollout_videos(targets, o_rollout_dummy, rewards, weight_dummy, 'Debug')