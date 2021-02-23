from predictors import *
from replay_memory_tools import extract_subtrajectories, trajectory_video, load_env_samples
from vae import *
#from vq_vae import *
from keras_vq_vae import VectorQuantizerEMAKeras
from blockworld import *
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import gym
import os
import sksfa
from sklearn.preprocessing import PolynomialFeatures
from replay_memory_tools import *
from matplotlib import pyplot as plt

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


def vq_vae_net(n_embeddings, d_embeddings, train_data_var, frame_stack=1):
    assert frame_stack == 1, 'No frame stacking supported currently'
    vae = VectorQuantizerEMAKeras(train_data_var, num_embeddings=n_embeddings, embedding_dim=d_embeddings)
    #vae = VqVAE(obs_shape, num_latent_k, latent_size, frame_stack=frame_stack, beta=1, kl_loss_factor_mul=1.000001)
    vae.compile(optimizer=tf.optimizers.Adam())
    return vae


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


def test_vae(vae, memories):
    all_observations = np.concatenate([np.append(mem['s'], mem['s_'][np.newaxis, -1], axis=0) for mem in memories], axis=0) / 255
    reconstructed = np.clip(vae.predict(all_observations), 0, 1)

    diff = np.abs(all_observations - reconstructed)
    plt.plot(diff.mean(axis=(1, 2, 3)))
    plt.show()


def gen_predictor_nets(vae, n_actions, num_heads):
    state_shape = vae.latent_shape
    per_head_filters = 64

    multihead_predictor = AutoregressiveMultiHeadFullyConvolutionalPredictor(state_shape, n_actions,
                                                                             per_head_filters=per_head_filters,
                                                                             n_heads=3)
    cb_map_shape = vae.enc_out_shape
    all_predictor = AutoregressiveProbabilisticFullyConvolutionalPredictor(cb_map_shape, n_actions, vae.codes_sampler,
                                                                           vae.n_cb_vectors, vae.latent_shape,
                                                                           vae.enc_out_shape,
                                                                           per_head_filters=per_head_filters * 3,
                                                                           n_models=1)
    complete_predictors_list = [multihead_predictor, all_predictor]
    for pred in complete_predictors_list:
        pred.compile(loss=['categorical_crossentropy', 'mse'],
                     optimizer=tf.optimizers.Adam(),
                     metrics={'output_1': tf.keras.metrics.CategoricalCrossentropy(),
                              'output_2': tf.keras.metrics.MeanSquaredError()})#, run_eagerly=True)

    return complete_predictors_list


def prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps):
    #obs = (trajectories['s'] / 255).astype(np.float32)
    #obs = cast_and_normalize_images(trajectories['s'])
    #next_obs = cast_and_normalize_images(trajectories['s_'])
    #next_obs = (trajectories['s_'] / 255).astype(np.float32)
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
    print('Dataset conversion done')

    # codebook matrix is shape (n, n), but for predictor we need (n, n, 1)
    #encoded_obs = np.expand_dims(encoded_obs, axis=-1)
    #encoded_next_obs = np.expand_dims(encoded_next_obs, axis=-1)

    encoded_obs = encoded_obs[:, 0:n_warmup_steps]
    encoded_next_obs = encoded_next_obs[:, :n_steps]
    action_inputs = actions[:, 0:n_steps, np.newaxis]
    reward_inputs = rewards[:, 0:n_steps, np.newaxis]

    return encoded_obs, encoded_next_obs, reward_inputs, action_inputs


def train_predictor(vae, predictor, trajectories, n_train_steps, n_traj_steps, n_warmup_steps,  predictor_weights_path, steps_per_epoch=200, batch_size=32):
    encoded_obs, encoded_next_obs, rewards, actions = prepare_predictor_data(trajectories, vae, n_traj_steps, n_warmup_steps)

    for i in range(len(encoded_obs) - 1):
        assert np.sum(encoded_obs[i + 1] - encoded_next_obs[i]) <= 0.0001

    dataset = (tf.data.Dataset.from_tensor_slices(((encoded_obs, actions), (encoded_next_obs, rewards)))
               .shuffle(steps_per_epoch * batch_size)
               .repeat(-1)
               .batch(batch_size, drop_remainder=True)
               .prefetch(-1))

    epochs = np.ceil(n_train_steps / steps_per_epoch).astype(np.int32)
    h = predictor.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    predictor.save_weights('predictors/' + predictor_weights_path)
    #with open('predictors/' + predictor_weights_path + '_train_stats', 'wb') as handle:
    #    pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return h

    #h = predictor.fit([env_idx, encoded_obs, actions],
    #                  [encoded_next_obs, rewards],
    #                  epochs=epochs,
    #                  batch_size=batch_size,
    #                  shuffle=True,
    #                  verbose=1,
    #                  validation_split=0.10)
    #
    #return h.history


def predictor_train_loop(complete_predictors_list, vae, memories, simulate_agent_memories, n_actions, n_epochs, n_subtrajectories, n_steps, n_warmup_steps):
    multihead_predictor = complete_predictors_list[0]
    all_predictor = complete_predictors_list[1]

    histories = [(n_epochs, n_subtrajectories, n_steps, n_warmup_steps)]
    #for i, mem in enumerate(simulate_agent_memories):
    #    print(f'Training multihead predictor\'s head {i + 1} on memory mix {i + 1} with time window {n_steps}, warmup {n_warmup_steps} and {n_epochs} epochs')
    #    trajs = extract_subtrajectories(mem, n_subtrajectories, n_steps + n_warmup_steps, False)
    #    env_idx = trajs['env'][:, 0]
    #    h = train_predictor(vae, multihead_predictor, trajs, env_idx, n_epochs, n_steps, n_warmup_steps)
    #    histories.append(h)
    for i, mem in enumerate(simulate_agent_memories):
        print(f'Training all-predictor on memory mix {i + 1} with time window {n_steps}, warmup {n_warmup_steps} and {n_epochs} epochs')
        trajs = extract_subtrajectories(mem, n_subtrajectories, n_steps + n_warmup_steps, False)
        env_idx = np.zeros_like(trajs['env'])
        h = train_predictor(vae, all_predictor, trajs, env_idx, n_epochs, n_steps, n_warmup_steps)
        histories.append(h)

    for i, pred in enumerate(complete_predictors_list):
        pred.save_weights(f'predictors/predictor_{i}')
    with open('predictors/train_stats', 'wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    #env_names = ['Gridworld-partial-room-v0', 'Gridworld-partial-room-v1', 'Gridworld-partial-room-v2']
    env_names = ['BoxingNoFrameskip-v0', 'SpaceInvadersNoFrameskip-v0', 'DemonAttackNoFrameskip-v0']
    obs_resize = (84, 84)
    collect_samples_per_env = 80000

    # vae params #
    n_vae_steps = 80000
    n_embeddings = 128
    d_embedding = 32
    frame_stack = 1

    # predictor params #
    n_pred_train_steps = 5000
    n_subtrajectories = 3000
    n_traj_steps = 20
    n_warmup_steps = 5

    tf.config.run_functions_eagerly(True)

    # start training procedure #

    sample_mem_paths = ['samples/raw/' + env_name for env_name in env_names]
    mix_mem_path = 'samples/mix/' + '_and_'.join(env_names) + '_mix'
    vae_weights_path = 'vae_model' + '_and_'.join(env_names) + '_vae_weights'
    predictor_weights_path = 'predictor_model' + '_and_'.join(env_names) + '_predictor_weights'

    #envs = [gym.make(env_name) for env_name in env_names]
    #envs = [gym.wrappers.GrayScaleObservation(gym.wrappers.ResizeObservation(gym.make(env_name), obs_resize), keep_dim=True) for env_name in env_names]
    envs = [gym.wrappers.AtariPreprocessing(gym.make(env_name), grayscale_newaxis=True) for env_name in env_names]
    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n

    #memories = gen_data(envs, samples_per_env=collect_samples_per_env, file_paths=sample_mem_paths)
    #gen_mixed_memory(memories, [1, 1, 1], file_path=mix_mem_path)
    #del memories

    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)

    vae = vq_vae_net(n_embeddings, d_embedding, train_data_var, frame_stack)
    vae_index_matrix_shape = vae.compute_latent_shape(mix_memory['s'])
    all_predictor = AutoregressiveProbabilisticFullyConvolutionalMultiHeadPredictor(vae_index_matrix_shape, n_actions,
                                                                                    vae,
                                                                                    open_loop_rollout_training=True,
                                                                                    det_filters=64,
                                                                                    prob_filters=64,
                                                                                    decider_lw=64,
                                                                                    n_models=1)#, debug_log=True)
    all_predictor.compile(optimizer=tf.optimizers.Adam())

    # train vae
    #load_vae_weights(vae, mix_memory, file_name=vae_weights_path, plots=False)
    #train_vae(vae, mix_memory, n_vae_steps, file_name=vae_weights_path, batch_size=512)
    load_vae_weights(vae, mix_memory, file_name=vae_weights_path, plots=True)

    # extract trajectories and train predictor
    trajs = extract_subtrajectories(mix_memory, n_subtrajectories, n_traj_steps, False)

    #all_predictor.load_weights('predictors/' + predictor_weights_path)
    train_predictor(vae, all_predictor, trajs, n_pred_train_steps, n_traj_steps, n_warmup_steps, predictor_weights_path, batch_size=50)
    all_predictor.load_weights('predictors/' + predictor_weights_path).expect_partial()

    generate_test_rollouts(all_predictor, mix_memory, vae, 100, 1, 'Predictor Test')


def generate_test_rollouts(predictor, mem, vae, n_steps, n_warmup_steps, video_title):
    n_trajectories = 5

    trajectories = extract_subtrajectories(mem, n_trajectories, n_steps, False)
    encoded_obs, _, _, actions = prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps)

    next_obs = trajectories['s_']
    targets = next_obs[:, n_warmup_steps - 1:]
    encoded_start_obs = encoded_obs[:, :n_warmup_steps]

    #print([env_idx.shape, encoded_start_obs.shape, one_hot_acts.shape])

    # do rollout
    o_rollout, r_rollout, w_predictors = predictor([encoded_start_obs, actions])

    #w_predictors = tf.transpose(w_predictors, [1, 2, 0]).numpy()

    chosen_predictor = np.argmax(tf.transpose(w_predictors, [1, 2, 0]), axis=-1)
    max_pred_idx = chosen_predictor.max()

    decoded_rollout_obs = cast_and_unnormalize_images(vae.decode_from_indices(o_rollout, 1))

    # remove batch dimension since it's 1 anyway
    all_videos = []
    all_titles = []
    for i, (ground_truth, rollout, pred_weight) in enumerate(zip(targets, decoded_rollout_obs, chosen_predictor)):
        weight_imgs = np.stack([np.full_like(ground_truth[0], w) / max_pred_idx * 255 for w in pred_weight])
        all_videos.extend([ground_truth, rollout, weight_imgs])
        all_titles.extend([f'true {i}', f'rollout {i}', f'weight {i}'])

    anim = trajectory_video(all_videos, all_titles, overall_title=video_title, max_cols=3)
    #writer = animation.writers['ffmpeg'](fps=10, bitrate=1800)
    #anim.save('rollout.mp4', writer=writer)


def vae_generalization_test():
    n_envs = 100
    envs = [gym.make('Gridworld-room-v0') for _ in range(n_envs)]
    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    latent_size = 32

    #memories = gen_data(envs, episodes=10, file_name='generalization_test_vae_train_data.npy')
    #simulate_agent_memories = gen_mixed_memories(memories, file_name='simulate_agent_memories.npy')
    memories = load_env_samples(['generalization_test_vae_train_data.npy'])[0]

    vae = vq_vae_net(obs_shape, latent_size)

    #load_vae_weights(vae, memories[-2:], plots=True)
    train_vae(vae, memories[:-1], steps=100)
    load_vae_weights(vae, memories, plots=True)
    #test_vae(vae, memories)

    sfa_transformer = sksfa.SFA(n_components=3)
    traj = extract_subtrajectories(memories[-1], 1, 200, False)[0]
    reconstructed = np.clip(vae.predict(traj['s'] / 255), 0, 1)
    #reconstructed = np.clip(vae.decode(data[..., np.newaxis]), 0, 1)
    #extracted_features = sfa_transformer.fit_transform(data)

    #plt.plot(extracted_features)
    #plt.show()
    anim = trajectory_video([traj['s'] / 255, reconstructed], ['true', 'reconstructed'])

    #train_predictors_2(all_predictors, vae, memories, simulate_agent_memories, n_actions, n_epochs=80, n_subtrajectories=15000, n_steps=5, n_warmup_steps=1)


def split_predictor_test():
    env_names = ['Gridworld-partial-room-v0', 'Gridworld-partial-room-v1', 'Gridworld-partial-room-v2']
    #env_names = ['Gridworld-partial-random-v0']
    envs = [gym.make(env_name) for env_name in env_names]

    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    latent_size = 20
    num_latent_k = 10

    vae = vq_vae_net(obs_shape, num_latent_k, latent_size)
    all_predictors = gen_predictor_nets(vae, n_actions, len(env_names))

    memories = gen_data(envs, samples_per_env=200, file_paths='vae_train_data.npy')
    simulate_agent_memories = gen_mixed_memory(memories, file_name='simulate_agent_memories.npy')
    memories, simulate_agent_memories = load_env_samples(['vae_train_data.npy', 'simulate_agent_memories.npy'])

    #train_pos_finder_net(pos_finder, memories, epochs=10)
    #load_pos_finder_weights(pos_finder, memories, pixel_size=2)

    load_vae_weights(vae, memories, plots=False)
    train_vae(vae, memories, steps=100)
    load_vae_weights(vae, memories, plots=False)

    #test_vae_parts(memories, vae)

    #load_predictor_weights(all_predictors, env_names, plots=True)
    predictor_train_loop(all_predictors, vae, memories, simulate_agent_memories, n_actions, n_epochs=10, n_subtrajectories=10000, n_steps=50, n_warmup_steps=1)
    load_predictor_weights(all_predictors, env_names, plots=True)

    #test_predictor(all_predictors[0], memories[0], True, vae, n_steps=200, n_warmup_steps=1, video_title='Predictor 0')
    #test_predictor(all_predictors[0], memories[1], True, vae, n_steps=200, n_warmup_steps=1, video_title='Predictor 1')
    #test_predictor(all_predictors[0], memories[2], True, vae, n_steps=200, n_warmup_steps=1, video_title='Predictor 2')

    generate_test_rollouts(all_predictors[1], memories[0], False, vae, n_steps=200, n_warmup_steps=1, video_title='All-Predictor Memory 0')
    generate_test_rollouts(all_predictors[1], memories[1], False, vae, n_steps=200, n_warmup_steps=1, video_title='All-Predictor Memory 0 + 1')
    generate_test_rollouts(all_predictors[1], memories[2], False, vae, n_steps=200, n_warmup_steps=1, video_title='All-Predictor Memory 0 + 1 + 2')

    n_steps = 500
    n_trajectories = 200
    #pred_0_perf = estimate_performance(all_predictors[0], memories[0], True, vae, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    #pred_1_perf = estimate_performance(all_predictors[0], memories[1], True, vae, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    #pred_2_perf = estimate_performance(all_predictors[0], memories[2], True, vae, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)

    all_pred_pref_0 = estimate_performance(all_predictors[1], memories[0], False, vae, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    all_pred_pref_1 = estimate_performance(all_predictors[1], memories[1], False, vae, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    all_pred_pref_2 = estimate_performance(all_predictors[1], memories[2], False, vae, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)

    #plt.plot(pred_0_perf, label='multihead_mem_0')
    #plt.plot(pred_1_perf, label='multihead_mem_1')
    #plt.plot(pred_2_perf, label='multihead_mem_2')
    plt.plot(all_pred_pref_0, label='all_predictor_mem_0', linestyle='--')
    plt.plot(all_pred_pref_1, label='all_predictor_mem_1', linestyle='--')
    plt.plot(all_pred_pref_2, label='all_predictor_mem_2', linestyle='--')
    plt.legend()
    plt.show()


def test_vae_parts(memories, vae):
    obs = memories[2]['s'][0:10000] / 255
    encodings = vae.encode(obs, 'indices')
    straight_through = vae.predict(obs)
    reconstructed = vae.decode(encodings, 'indices')
    diff_reconstructed = np.abs(obs - reconstructed)
    diff_straight_thorugh = np.abs(obs - straight_through)
    print(f'Difference reconstructed: {np.mean(diff_reconstructed)}\nDifference straight through: {np.mean(diff_straight_thorugh)}')
    #trajectory_video([obs, straight_through, reconstructed], ['obs', 'straight through', 'reconstructed'])
    #plt.show()


def estimate_performance(predictor, mem, use_env_idx, vae, n_steps, n_warmup_steps, n_trajectories):
    predictor.predict_steps = n_steps

    trajectories = extract_subtrajectories(mem, n_trajectories, n_steps + n_warmup_steps - 1, False)
    encoded_obs, _, _, actions = prepare_predictor_data(trajectories, vae, n_steps, n_warmup_steps)

    next_obs = trajectories['s_'] / 255
    targets = next_obs[:, 0:n_steps]
    encoded_start_obs = encoded_obs[:, 0:n_warmup_steps]

    if use_env_idx:
        env_idx = trajectories['env'][:, 0]
    else:
        env_idx = np.zeros_like(trajectories['env'][:, 0])

    # do rollout
    rollout = predictor.predict([env_idx, encoded_start_obs, actions], n_steps)
    decoded_rollout_obs = np.clip(vae.decode(rollout, 'indices'), 0, 1)

    # find player pos
    #analyzed_player_pos = pos_finder.predict(decoded_rollout_obs).astype(int)
    #pos_diff = np.abs(end_pos - analyzed_player_pos)
    #pos_diff /= 2
    # debug_img = np.repeat(pos_diff, 3, axis=1).reshape(-1, 1, 2, 3)
    diff_img = np.clip(np.abs(decoded_rollout_obs - targets), 0, 1)

    return diff_img.mean(axis=(0, 2, 3, 4))


def distibution_test():
    import tensorflow_probability as tfp
    from tensorflow_probability import distributions as tfd

    n_batch = 32
    n_cat = 10
    logits = np.zeros((n_batch, 5, 5, n_cat))
    logits[:, :, 0] = 1
    #logits = [-1.1, 0.8, 0.1]
    dist = tfd.RelaxedOneHotCategorical(logits=logits, temperature=0.001)
    print(dist)

    layer = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Independent(tfd.RelaxedOneHotCategorical(0.001, logits=logits),
                                                       reinterpreted_batch_ndims=2),
        convert_to_tensor_fn=lambda t: t.sample(),
    )
    distribution = layer(None)
    print(distribution)
    #print(distribution.sample())
    #print(distribution.sample())
    #print(distribution.sample())

    return

    n_x = 200
    n_c = 20
    output_shape = (5, 5)

    x = np.random.randint(0, n_c, 200).astype(np.float32)
    y = [np.full((5, 5), val).astype(np.float32) for val in x]
    y[:, 0, 0] = 1

    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    inp = tf.keras.layers.Input((1,))
    layer = tf.keras.layers.Dense(5 * 5 * tfp.layers.OneHotCategorical.params_size(n_c))(inp)
    layer = tf.keras.layers.Reshape((5, 5, tfp.layers.OneHotCategorical.params_size(n_c)))(layer)
    #layer = tfp.layers.DistributionLambda(
    #    make_distribution_fn=lambda t: tfd.RelaxedOneHotCategorical(0.0005, logits=t),
    #    convert_to_tensor_fn=lambda t: t.sample(),
    #)(layer)
    layer = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.RelaxedOneHotCategorical(0.001, logits=t),
        convert_to_tensor_fn=lambda t: t.sample(),
    )(layer)
    layer = OneHotToIndex()(layer)
    #layer = tf.keras.layers.Lambda(lambda t: tf.matmul(tf.repeat(tf.range(t.get_shape()[-1], dtype=tf.float32)), t))(layer)
    mdl = keras.Model(inputs=inp, outputs=layer)
    #mdl.summary()

    # Do inference.
    mdl.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')
    mdl.fit(x, y, epochs=100, verbose=False)

    y_hat = mdl.predict([0, 1])
    print(y_hat)


if __name__ == '__main__':
    #distibution_test()
    #vae_generalization_test()
    #split_predictor_test()
    single_predictor()

    #inp = np.array([[0, 1, 0], [1, 1, 0]])
    #print(inp.shape)
    #l = InflateActionLayer((5, 5), 3)
    #outp = l(inp)
    #print(outp)
