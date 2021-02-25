from old_predictors import AutoregressiveMultiHeadFullyConvolutionalPredictor, \
    AutoregressiveProbabilisticFullyConvolutionalPredictor
from predictors import *
#from vq_vae import *
from keras_vq_vae import VectorQuantizerEMAKeras
from replay_memory_tools import *
from matplotlib import pyplot as plt
from blockworld import *

#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

# see https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def vq_vae_net(obs_shape, n_embeddings, d_embeddings, train_data_var, grayscale_input=False, frame_stack=1):
    assert frame_stack == 1, 'No frame stacking supported currently'
    vae = VectorQuantizerEMAKeras(train_data_var, num_embeddings=n_embeddings, embedding_dim=d_embeddings,
                                  grayscale_input=grayscale_input)
    vae.compile(optimizer=tf.optimizers.Adam())
    vae.build((None, *obs_shape))
    return vae


def predictor_net(n_actions, obs_shape, vae):
    vae_index_matrix_shape = vae.compute_latent_shape(obs_shape)
    all_predictor = AutoregressiveProbabilisticFullyConvolutionalMultiHeadPredictor(vae_index_matrix_shape, n_actions,
                                                                                    vae,
                                                                                    open_loop_rollout_training=True,
                                                                                    det_filters=16,
                                                                                    prob_filters=16,
                                                                                    decider_lw=16,
                                                                                    n_models=3)  # , debug_log=True)
    all_predictor.compile(optimizer=tf.optimizers.Adam())
    net_s_obs = tf.TensorShape((None, None, *vae.compute_latent_shape(obs_shape)))
    net_s_act = tf.TensorShape((None, None, 1))
    all_predictor.build([net_s_obs, net_s_act])
    return all_predictor


def train_vae(vae, memory, steps, file_name, batch_size=256, steps_per_epoch=200):
    if vae.frame_stack == 1:
        all_observations = line_up_observations(memory)
    else:
        all_observations = stack_observations(memory, vae.frame_stack)
    print('Total number of training samples: {}'.format(len(all_observations)))

    train_dataset = (tf.data.Dataset.from_tensor_slices(all_observations)
                     .map(cast_and_normalize_images)
                     .shuffle(steps_per_epoch * batch_size)
                     .repeat(-1)  # repeat indefinitely
                     .batch(batch_size, drop_remainder=True)
                     .prefetch(-1))

    #history = vae.fit(train_dataset, epochs=steps, verbose=1, batch_size=batch_size, shuffle=True, validation_split=0.1).history
    epochs = np.ceil(steps / steps_per_epoch).astype(np.int32)
    history = vae.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1).history

    vae.save_weights('vae_model/' + file_name)
    with open('vae_model/' + file_name + '_train_stats', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps):
    actions = trajectories['a'].astype(np.int32)
    rewards = trajectories['r'].astype(np.float32)

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

    return encoded_obs, encoded_next_obs, reward_inputs, action_inputs


def train_predictor(vae, predictor, trajectories, n_train_steps, n_traj_steps, n_warmup_steps,  predictor_weights_path, steps_per_epoch=200, batch_size=32):
    encoded_obs, encoded_next_obs, rewards, actions = prepare_predictor_data(trajectories, vae, n_traj_steps, n_warmup_steps)

    # sanity check
    for i_t in range(n_warmup_steps - 1):
        assert np.isclose(encoded_obs[:, i_t + 1], encoded_next_obs[:, i_t], atol=0.001).all(), 'Trajectory seems to be corrupted'

    dataset = (tf.data.Dataset.from_tensor_slices(((encoded_obs, actions), (encoded_next_obs, rewards)))
               .shuffle(100000)
               .repeat(-1)
               .batch(batch_size, drop_remainder=True)
               .prefetch(-1))

    epochs = np.ceil(n_train_steps / steps_per_epoch).astype(np.int32)
    h = predictor.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    predictor.save_weights('predictors/' + predictor_weights_path)
    #with open('predictors/' + predictor_weights_path + '_train_stats', 'wb') as handle:
    #    pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return h


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


def single_predictor():
    # env #
    env_names = ['Gridworld-partial-room-v0', 'Gridworld-partial-room-v1', 'Gridworld-partial-room-v2']
    #env_names = ['BoxingNoFrameskip-v0', 'SpaceInvadersNoFrameskip-v0', 'DemonAttackNoFrameskip-v0']
    obs_resize = (84, 84)
    grayscale_input = False
    collect_samples_per_env = 50000

    # vae params #
    n_vae_steps = 40000
    n_embeddings = 128
    d_embedding = 32
    frame_stack = 1

    # predictor params #
    n_pred_train_steps = 10000
    n_subtrajectories = 5000
    n_traj_steps = 10
    n_warmup_steps = 5

    #tf.config.run_functions_eagerly(True)

    # start training procedure #

    sample_mem_paths = ['samples/raw/' + env_name for env_name in env_names]
    mix_mem_path = 'samples/mix/' + '_and_'.join(env_names) + '_mix'
    vae_weights_path = 'vae_model' + '_and_'.join(env_names) + '_vae_weights'
    predictor_weights_path = 'predictor_model' + '_and_'.join(env_names) + '_predictor_weights'

    envs = [gym.make(env_name) for env_name in env_names]
    #envs = [gym.wrappers.GrayScaleObservation(gym.wrappers.ResizeObservation(gym.make(env_name), obs_resize), keep_dim=True) for env_name in env_names]
    #envs = [gym.wrappers.AtariPreprocessing(gym.make(env_name), grayscale_newaxis=True) for env_name in env_names]
    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n

    #memories = gen_data(envs, samples_per_env=collect_samples_per_env, file_paths=sample_mem_paths)
    #gen_mixed_memory(memories, [1, 1, 1], file_path=mix_mem_path)
    #del memories

    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)

    vae = vq_vae_net(obs_shape, n_embeddings, d_embedding, train_data_var, grayscale_input, frame_stack)
    #vae.summary()
    all_predictor = predictor_net(n_actions, obs_shape, vae)
    #all_predictor.summary()

    # train vae
    #load_vae_weights(vae, mix_memory, file_name=vae_weights_path, plots=False)
    #train_vae(vae, mix_memory, n_vae_steps, file_name=vae_weights_path, batch_size=512)
    load_vae_weights(vae, mix_memory, file_name=vae_weights_path, plots=True)

    # extract trajectories and train predictor
    trajs = extract_subtrajectories(mix_memory, n_subtrajectories, n_traj_steps, False)
    #all_predictor.load_weights('predictors/' + predictor_weights_path)
    train_predictor(vae, all_predictor, trajs, n_pred_train_steps, n_traj_steps, n_warmup_steps, predictor_weights_path, batch_size=32)
    all_predictor.load_weights('predictors/' + predictor_weights_path).expect_partial()
    all_predictor.summary()

    targets, rollouts, w_predictors = generate_test_rollouts(all_predictor, mix_memory, vae, 200, 1, 200)
    #rollout_videos(targets, rollouts, w_predictors, 'Predictor Test')

    plt.hist(np.array(w_predictors).flatten())
    plt.show()

    pixel_diff_mean = np.mean(targets - rollouts, axis=(0, 2, 3, 4))
    pixel_diff_var = np.std(targets - rollouts, axis=(0, 2, 3, 4))
    x = range(len(pixel_diff_mean))
    plt.plot(x, pixel_diff_mean)
    plt.fill_between(x, pixel_diff_mean - pixel_diff_var, pixel_diff_mean + pixel_diff_var, alpha=0.2)
    plt.show()


def generate_test_rollouts(predictor, mem, vae, n_steps, n_warmup_steps, n_trajectories):
    if not predictor.open_loop_rollout_training:
        n_warmup_steps = n_steps

    trajectories = extract_subtrajectories(mem, n_trajectories, n_steps, False)
    encoded_obs, _, _, actions = prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps)

    next_obs = trajectories['s_']
    if not predictor.open_loop_rollout_training:
        targets = next_obs[:, :n_warmup_steps]
    else:
        targets = next_obs[:, n_warmup_steps - 1:]
    encoded_start_obs = encoded_obs[:, :n_warmup_steps]

    #print([env_idx.shape, encoded_start_obs.shape, one_hot_acts.shape])

    # do rollout
    o_rollout, r_rollout, w_predictors = predictor([encoded_start_obs, actions])
    chosen_predictor = np.argmax(tf.transpose(w_predictors, [1, 2, 0]), axis=-1)
    decoded_rollout_obs = cast_and_unnormalize_images(vae.decode_from_indices(o_rollout)).numpy()

    return targets, decoded_rollout_obs, chosen_predictor


def rollout_videos(targets, decoded_rollout_obs, chosen_predictor, video_title, store_animation=False):
    max_pred_idx = chosen_predictor.max() + 1e-5
    all_videos = []
    all_titles = []
    for i, (ground_truth, rollout, pred_weight) in enumerate(zip(targets, decoded_rollout_obs, chosen_predictor)):
        weight_imgs = np.stack([np.full_like(ground_truth[0], w) / max_pred_idx * 255 for w in pred_weight])
        all_videos.extend([ground_truth, rollout, weight_imgs])
        all_titles.extend([f'true {i}', f'rollout {i}', f'weight {i}'])
    anim = trajectory_video(all_videos, all_titles, overall_title=video_title, max_cols=3)

    if store_animation:
        writer = animation.writers['ffmpeg'](fps=10, bitrate=1800)
        anim.save('rollout.mp4', writer=writer)


if __name__ == '__main__':
    single_predictor()