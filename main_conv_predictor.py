from predictors import *
from keras_vq_vae import VectorQuantizerEMAKeras
from replay_memory_tools import *
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from blockworld import *
import gym_minigrid
import tensorflow_probability as tfp
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

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

    run = neptune.init(project='rschiewer/predictor')
    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    run['parameters'] = {'n_train_steps': steps, 'vqave_weights_path': file_name}
    run['sys/tags'].add('vqvae')

    #history = vae.fit(train_dataset, epochs=steps, verbose=1, batch_size=batch_size, shuffle=True, validation_split=0.1).history
    epochs = np.ceil(steps / steps_per_epoch).astype(np.int32)
    history = vae.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[neptune_cbk]).history

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


def train_predictor(vae, predictor, trajectories, n_train_steps, n_traj_steps, n_warmup_steps,  predictor_weights_path, steps_per_epoch=200, batch_size=32):
    encoded_obs, encoded_next_obs, rewards, actions, terminals = prepare_predictor_data(trajectories, vae, n_traj_steps, n_warmup_steps)

    # sanity check
    #for i_t in range(n_warmup_steps - 1):
    #    if not np.isclose(encoded_obs[:, i_t + 1], encoded_next_obs[:, i_t], atol=0.001).all():
    #        wrong_index = np.argmax(np.abs(np.sum(encoded_next_obs[:, i_t] - encoded_obs[:, i_t + 1], axis=(1, 2))))
    #        fig, (ax0, ax1) = plt.subplots(1, 2)
    #        ax0.imshow(trajectories[wrong_index]['s'][i_t + 1])
    #        ax1.imshow(trajectories[wrong_index]['s_'][i_t])
    #        plt.show()
    #        print(f'Trajectory seems to be corrupted {np.sum(encoded_obs[:, i_t + 1] - encoded_next_obs[:, i_t])}')
    #for i_traj, (traj, r) in enumerate(zip(trajectories, rewards)):
    #    if True in traj['done']:
    #        terminals = np.nonzero(traj['done'])
    #        assert len(terminals) == 1
    #        if r[terminals[0]] == 1:
    #            plt.imshow(trajectories[i_traj][terminals[0]]['s'][0])
    #            plt.show()

    dataset = (tf.data.Dataset.from_tensor_slices(((encoded_obs, actions), (encoded_next_obs, rewards, terminals)))
               .shuffle(50000)
               .repeat(-1)
               .batch(batch_size, drop_remainder=True)
               .prefetch(-1))

    run = neptune.init(project='rschiewer/predictor')
    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    run['parameters'] = {'n_train_steps': n_train_steps, 'n_traj_steps': n_traj_steps,
                         'n_warmup_steps': n_warmup_steps, 'predictor_weights_path': predictor_weights_path,
                         'det_filters': predictor.det_filters, 'prob_filters': predictor.prob_filters,
                         'decider_lw': predictor.decider_lw, 'n_models': predictor.n_models}

    run['sys/tags'].add('predictor')

    epochs = np.ceil(n_train_steps / steps_per_epoch).astype(np.int32)
    h = predictor.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[neptune_cbk]).history
    predictor.save_weights('predictors/' + predictor_weights_path)
    with open('predictors/' + predictor_weights_path + '_train_stats', 'wb') as handle:
        pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

def plan(predictor, vae, start_sample, n_actions, plan_steps, n_rollouts, n_iterations=10, top_perc=0.1):
    """Crossentropy method, see algorithm 2.2 from https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf,
    https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
    and https://towardsdatascience.com/cross-entropy-method-for-reinforcement-learning-2b6de2a4f3a0
    """
    # add axis for batch dim when encoding
    encoded_start_sample = vae.encode_to_indices(start_sample[tf.newaxis, ...])
    # add axis for time, then repeat n_rollouts times along batch dimension
    o_in = tf.repeat(encoded_start_sample[tf.newaxis, ...], repeats=[n_rollouts], axis=0)
    # initial params for sampling distribution
    dist_params = tf.ones((plan_steps, n_actions), dtype=tf.float32) / n_actions
    k = tf.cast(tf.round(n_rollouts * top_perc), tf.int32)

    for i_iter in range(n_iterations):
        # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
        # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
        a_in = tfp.distributions.Categorical(probs=dist_params).sample(n_rollouts)
        a_in = tf.expand_dims(a_in, axis=-1)

        o_pred, r_pred, done_pred, pred_weights = predictor([o_in, a_in])
        r_pred = np.squeeze(r_pred.numpy())

        # make sure trajectory ends after reward was collected once
        processed_r_pred = np.zeros_like(r_pred)
        for i_traj in range(len(r_pred)):
            if tf.reduce_sum(r_pred[i_traj]) > 1.0:
                i_first_reward = np.min(np.nonzero(r_pred[i_traj] > 0.4))
                processed_r_pred[i_traj, 0: i_first_reward + 1] = r_pred[i_traj, 0: i_first_reward + 1]
            else:
                processed_r_pred[i_traj] = r_pred[i_traj]

        #returns = tf.reduce_sum(processed_r_pred, axis=1)

        # discounted returns to prefer shorter trajectories
        discounted_returns = tf.map_fn(
            lambda r_trajectory: tf.scan(lambda cumsum, elem: cumsum + 0.9 * elem, r_trajectory)[-1],
            processed_r_pred
        )

        top_returns, top_i_a_sequence = tf.math.top_k(discounted_returns, k=k)
        top_a_sequence = tf.gather(a_in, top_i_a_sequence)

        print(f'Top returns are: {top_returns}')
        #trajectory_video(cast_and_unnormalize_images(vae.decode_from_indices(o_pred[top_i_a_sequence[0], tf.newaxis, ...])), ['best sequence'])

        # MLE for categorical, see
        # https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
        # here we have multiple samples for MLE, which means the parameter update for one timestep is:
        # theta_i = sum_k a_ki / (sum_i sum_k a_ki) with i=action_index, k=sample
        top_a_sequence_onehot = tf.one_hot(top_a_sequence, n_actions, axis=-1)[:, :, 0, :]  # remove redundant dim
        numerator = tf.reduce_sum(top_a_sequence_onehot, axis=0)
        denominator = tf.reduce_sum(top_a_sequence_onehot, axis=[0, 2])[..., tf.newaxis]
        dist_params = numerator / denominator

    print(f'Final action probabilities: {dist_params[0]}')
    return tf.argmax(dist_params, axis=1)
    #return tfp.distributions.Categorical(probs=dist_params).sample()


def plan_gaussian(predictor, vae, start_sample, n_actions, plan_steps, n_rollouts, n_iterations=10, top_perc=0.1):
    """Crossentropy method, see algorithm 2.2 from https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf
    """
    # add axis for batch dim when encoding
    encoded_start_sample = vae.encode_to_indices(start_sample[tf.newaxis, ...])
    # add axis for time, then repeat n_rollouts times along batch dimension
    o_in = tf.repeat(encoded_start_sample[tf.newaxis, ...], repeats=[n_rollouts], axis=0)
    mean = tf.random.uniform((plan_steps,), minval=0, maxval=n_actions - 1, dtype=tf.float32)
    scale = tf.random.uniform((plan_steps,), dtype=tf.float32)
    k = tf.cast(tf.round(n_rollouts * top_perc), tf.int32)

    for i_iter in range(n_iterations):
        # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
        # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
        a_in = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale).sample(n_rollouts)
        a_in = tf.cast(tf.round(a_in), tf.int32)
        a_in = tf.clip_by_value(a_in, 0, n_actions - 1)
        a_in = tf.expand_dims(a_in, axis=-1)

        o_pred, r_pred, pred_weights = predictor([o_in, a_in])
        r_pred = np.squeeze(r_pred.numpy())

        # make sure trajectory ends after reward was collected once
        processed_r_pred = np.zeros_like(r_pred)
        for i_traj in range(len(r_pred)):
            if tf.reduce_sum(r_pred[i_traj]) > 1.0:
                i_first_reward = np.min(np.nonzero(r_pred[i_traj] > 0.75))
                processed_r_pred[i_traj, 0: i_first_reward + 1] = r_pred[i_traj, 0: i_first_reward + 1]
            else:
                processed_r_pred[i_traj] = r_pred[i_traj]

        returns = tf.reduce_sum(processed_r_pred, axis=1)

        # discounted returns to prefer shorter trajectories
        discounted_returns = tf.map_fn(
            lambda r_trajectory: tf.scan(lambda cumsum, elem: cumsum + 0.9 * elem, r_trajectory)[-1],
            processed_r_pred
        )

        top_returns, top_i_a_sequence = tf.math.top_k(discounted_returns, k=k)
        top_a_sequence = tf.gather(a_in, top_i_a_sequence)[:, :, 0]
        top_a_sequence = tf.cast(top_a_sequence, tf.float64)

        print(f'Top returns are: {top_returns}')
        #trajectory_video(cast_and_unnormalize_images(vae.decode_from_indices(o_pred[top_i_a_sequence[0], tf.newaxis, ...])), ['best sequence'])

        mean, scale = tf.nn.moments(top_a_sequence, axes=[0])

    print(f'Final mean: {mean}')
    print(f'Final var: {scale}')
    return tf.cast(tf.round(tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=scale).sample()), tf.int32)


def control(predictor, vae, env, env_info, plan_steps=50, n_rollouts=64, n_iterations=5, top_perc=0.1, do_mpc=True, render=False):
    last_observation = env.reset()
    t = 0
    available_actions = []
    while True:
        if render:
            env.render()

        if len(available_actions) == 0:
            obs_preprocessed = cast_and_normalize_images(last_observation)
            actions = plan(predictor, vae, obs_preprocessed, env_info['n_actions'], plan_steps, n_rollouts,
                           n_iterations, top_perc)
            available_actions.extend([a for a in actions.numpy()])
        action = available_actions.pop(0)
        if do_mpc:
            available_actions.clear()
        act_names = ['up', 'right', 'down', 'left', 'noop']
        print(f'action: {act_names[action]}')
        observation, reward, done, info = env.step(action)

        if done:
            break
        else:
            last_observation = observation
        t += 1
    print(f'Environment solved within {t} steps.')
    env.close()


def train_routine():
    # general settings #
    tensorflow_eager_mode = False
    model_summaries = False
    predictor_tensorboard_log = False

    # env #
    env_names, envs, env_info = gen_environments('gridworld_3_rooms')
    collect_samples_per_env = 80000

    # vae params #
    n_vae_steps = 40000
    commitment_cost = 0.25
    n_embeddings = 64
    d_embedding = 32
    frame_stack = 1

    # predictor params #
    n_pred_train_steps = 10000
    n_subtrajectories = 5000
    n_traj_steps = 15
    n_warmup_steps = 5
    pad_trajectories = True
    det_filters = 32
    prob_filters = 32
    decider_lw = 64
    n_models = 4
    pred_batch_size = 64


    # start training procedure #

    tf.config.run_functions_eagerly(tensorflow_eager_mode)

    sample_mem_paths = ['samples/raw/' + env_name for env_name in env_names]
    mix_mem_path = 'samples/mix/' + '_and_'.join(env_names) + '_mix'
    vae_weights_path = 'vae_model' + '_and_'.join(env_names) + '_vae_weights'
    predictor_weights_path = 'predictor_model' + '_and_'.join(env_names) + '_predictor_weights' + '_' + str(n_models) + '_models'

    #memories = gen_data(envs, env_info, samples_per_env=collect_samples_per_env, file_paths=sample_mem_paths)
    #gen_mixed_memory(memories, [1, 1, 1], file_path=mix_mem_path)
    #del memories

    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)

    vae = vq_vae_net(env_info['obs_shape'], n_embeddings, d_embedding, train_data_var, commitment_cost, frame_stack,
                     summary=model_summaries)
    all_predictor = predictor_net(env_info['n_actions'], env_info['obs_shape'], vae, det_filters, prob_filters,
                                  decider_lw, n_models, predictor_tensorboard_log, summary=model_summaries)

    #rewards = cumulative_episode_rewards(mix_memory)
    #rewards_from_mem = mix_memory['r'].sum()
    #plt.plot(rewards, label='cumulative episode rewards')
    #plt.show()

    # train vae
    #load_vae_weights(vae, mix_memory, file_name=vae_weights_path, plots=False)
    #train_vae(vae, mix_memory, n_vae_steps, file_name=vae_weights_path, batch_size=512)
    load_vae_weights(vae, mix_memory, file_name=vae_weights_path, plots=False)

    # extract trajectories and train predictor
    trajs = extract_subtrajectories(mix_memory, n_subtrajectories, n_traj_steps, pad_short_trajectories=pad_trajectories)
    #all_predictor.load_weights('predictors/' + predictor_weights_path)
    train_predictor(vae, all_predictor, trajs, n_pred_train_steps, n_traj_steps, n_warmup_steps, predictor_weights_path, batch_size=pred_batch_size)
    all_predictor.load_weights('predictors/' + predictor_weights_path)

    #_debug_visualize_trajectory(trajs)

    #predictor_allocation_stability(all_predictor, mix_memory, vae, 0)
    #predictor_allocation_stability(all_predictor, mix_memory, vae, 1)
    #predictor_allocation_stability(all_predictor, mix_memory, vae, 2)
    #quit()

    targets, o_rollout, r_rollout, done_rollout, w_predictors = generate_test_rollouts(all_predictor, mix_memory, vae, 200, 10, 4)
    rollout_videos(targets, o_rollout, r_rollout, done_rollout, w_predictors, 'Predictor Test')

    # rewards
    for i, r_traj in enumerate(r_rollout):
        plt.plot(np.squeeze(r_traj), label=f'reward rollout {i}')
    plt.legend()
    plt.show()

    # terminal probabilities
    for i, done_traj in enumerate(done_rollout):
        plt.plot(np.squeeze(done_traj), label=f'terminal prob rollout {i}')
    plt.legend()
    plt.show()

    # predictor choice
    plt.hist(np.array(w_predictors).flatten())
    plt.show()

    # difference between predicted and true observations
    pixel_diff_mean = np.mean(targets - o_rollout, axis=(0, 2, 3, 4))
    pixel_diff_var = np.std(targets - o_rollout, axis=(0, 2, 3, 4))
    x = range(len(pixel_diff_mean))
    plt.plot(x, pixel_diff_mean)
    plt.fill_between(x, pixel_diff_mean - pixel_diff_var, pixel_diff_mean + pixel_diff_var, alpha=0.2)
    plt.show()

    control(all_predictor, vae, envs[1], env_info, render=True, plan_steps=100, n_iterations=3, n_rollouts=200, top_perc=0.1, do_mpc=False)


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


if __name__ == '__main__':
    train_routine()