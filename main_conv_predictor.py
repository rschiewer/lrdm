from train_tools import gen_environments, vq_vae_net, predictor_net, prepare_predictor_data, load_vae_weights, \
    generate_test_rollouts, rollout_videos
from replay_memory_tools import *
from matplotlib import pyplot as plt
from blockworld import *
import tensorflow_probability as tfp
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from project_init import CONFIG

#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'


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

    logs = 'tb_profile_log'
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='100, 120')

    epochs = np.ceil(n_train_steps / steps_per_epoch).astype(np.int32)
    h = predictor.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[neptune_cbk]).history
    predictor.save_weights('predictors/' + predictor_weights_path)
    with open('predictors/' + predictor_weights_path + '_train_stats', 'wb') as handle:
        pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return h


def plan(predictor, vae, start_sample, n_actions, plan_steps, n_rollouts, n_iterations, top_perc, gamma):
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

    assert n_iterations > 0, f'Number of iterations must be geater than 0 but is {n_iterations}'

    for i_iter in range(n_iterations):
        # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
        # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
        a_in = tfp.distributions.Categorical(probs=dist_params).sample(n_rollouts)
        a_in = tf.expand_dims(a_in, axis=-1)

        o_pred, r_pred, done_pred, pred_weights = predictor([o_in, a_in])

        # make sure trajectory ends after reward was collected once
        #processed_r_pred = np.zeros_like(r_pred)
        #for i_traj in range(len(r_pred)):
        #    if tf.reduce_sum(r_pred[i_traj]) > 1.0:
        #        i_first_reward = np.min(np.nonzero(r_pred[i_traj] > 0.4))
        #        processed_r_pred[i_traj, 0: i_first_reward + 1] = r_pred[i_traj, 0: i_first_reward + 1]
        #    else:
        #        processed_r_pred[i_traj] = r_pred[i_traj]
        done_pred_prepend_dummy = tf.concat([tf.zeros((n_rollouts, 1), dtype=tf.float32), done_pred[:, :-1, 0]], axis=1)
        discount_factors = tf.map_fn(
            lambda d_traj: tf.scan(lambda cumulative, elem: cumulative * gamma * (1 - elem), d_traj, initializer=1.0),
            done_pred_prepend_dummy
        )

        discounted_returns = tf.reduce_sum(discount_factors * tf.squeeze(r_pred), axis=1)
        #returns = tf.reduce_sum(processed_r_pred, axis=1)

        # discounted returns to prefer shorter trajectories
        #discounted_returns = tf.map_fn(
        #    lambda r_trajectory: tf.scan(lambda cumsum, elem: cumsum + elem, r_trajectory)[-1],
        #    r_pred * discount_factors
        #)

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
    return top_a_sequence[0, :, 0]  # take best guess from last iteration and remove redundant dimension


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


def control(predictor, vae, env, env_info, plan_steps=50, n_rollouts=64, n_iterations=5, top_perc=0.1, gamma=0.99,
            do_mpc=True, render=False):
    last_observation = env.reset()
    t = 0
    available_actions = []
    while True:
        if render:
            env.render()

        if len(available_actions) == 0:
            obs_preprocessed = cast_and_normalize_images(last_observation)
            actions = plan(predictor, vae, obs_preprocessed, env_info['n_actions'], plan_steps, n_rollouts,
                           n_iterations, top_perc, gamma)
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
    prob_filters = 64
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

    #control(all_predictor, vae, envs[2], env_info, render=True, plan_steps=250, n_iterations=5, n_rollouts=250,
    #        top_perc=0.05, gamma=0.95, do_mpc=True)

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


if __name__ == '__main__':
    train_routine()