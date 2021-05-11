import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from replay_memory_tools import cast_and_normalize_images
from tools import ValueHistory


def plan(predictor, prev_dist_params, preprocessed_start_samples, preprocessed_start_actions, n_actions, plan_steps, n_rollouts,
         n_iterations, top_perc, gamma, action_noise, env_name, neptune_run, debug_plot=False):
    """Crossentropy method, see algorithm 2.2 from https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf,
    https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
    and https://towardsdatascience.com/cross-entropy-method-for-reinforcement-learning-2b6de2a4f3a0
    """

    # add axis for batch dim when encoding
    # add axis for batch, then repeat n_rollouts times along batch dimension
    o_hist = tf.repeat(preprocessed_start_samples[tf.newaxis, ...], repeats=[n_rollouts], axis=0)
    a_hist = tf.repeat(preprocessed_start_actions[tf.newaxis, :, tf.newaxis], repeats=[n_rollouts], axis=0)
    # initial params for sampling distribution
    if prev_dist_params is None:
        dist_params = tf.ones((plan_steps, n_actions), dtype=tf.float32) / n_actions
    else:
        dist_params = prev_dist_params
    k = tf.cast(tf.round(n_rollouts * top_perc), tf.int32)

    assert n_iterations > 0, f'Number of iterations must be geater than 0 but is {n_iterations}'

    per_predictor_weights = []
    if debug_plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6))
    for i_iter in range(n_iterations):
        # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
        # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
        a_new = tfp.distributions.Categorical(probs=dist_params).sample(n_rollouts)[..., tf.newaxis]
        a_in = tf.concat([a_hist, a_new], axis=1)

        o_pred, r_pred, done_pred, pred_weights = predictor([o_hist, a_in])

        # make sure trajectory ends after reward was collected once
        done_mask = tf.concat([tf.zeros((n_rollouts, 1), dtype=tf.float32), done_pred[:, :-1, 0]], axis=1)
        discount_factors = tf.map_fn(
            lambda d_traj: tf.scan(lambda cumulative, elem: cumulative * gamma * (1 - elem), d_traj, initializer=1.0),
            done_mask
        )

        discounted_returns = tf.reduce_sum(discount_factors * r_pred[:, :, 0], axis=1)
        #returns = tf.reduce_sum(processed_r_pred, axis=1)

        top_returns, top_i_a_sequence = tf.math.top_k(discounted_returns, k=k)
        top_a_sequence = tf.gather(a_new, top_i_a_sequence)
        top_dones = tf.gather(done_pred[:, :, 0], top_i_a_sequence)

        if debug_plot:
            done_idxs = tf.math.argmax(top_dones, axis=1)
            term_gammas = tf.gather(top_dones, done_idxs[..., tf.newaxis], batch_dims=1)
            traj_lengths = tf.where(term_gammas[:, 0] > 0.9, done_idxs + 1, plan_steps)
            print(f'Top returns are: {top_returns}')
            print(f'Trajectory lengths: {traj_lengths}')
            ax0.plot(top_returns.numpy(), label=f'iteration_{i_iter}')
            ax1.plot(traj_lengths.numpy(), label=f'iteration_{i_iter}')
            #print(f'Top first action: {top_a_sequence[0, 0, 0]}')
            #trajectory_video(cast_and_unnormalize_images(vae.decode_from_indices(o_pred[top_i_a_sequence[0], tf.newaxis, ...])), ['best sequence'])

        # MLE for categorical, see
        # https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
        # here we have multiple samples for MLE, which means the parameter update for one timestep is:
        # theta_i = sum_k a_ki / (sum_i sum_k a_ki) with i=action_index, k=sample
        top_a_sequence_onehot = tf.one_hot(top_a_sequence, n_actions, axis=-1)[:, :, 0, :]  # remove redundant dim
        numerator = tf.reduce_sum(top_a_sequence_onehot, axis=0)
        denominator = tf.reduce_sum(top_a_sequence_onehot, axis=[0, 2])[..., tf.newaxis]
        dist_params = numerator / denominator

        if action_noise:
            dist_params += tf.random.normal(tf.shape(dist_params), 0, action_noise)
            dist_params = tf.clip_by_value(dist_params, 0, 1)
            dist_params /= tf.reduce_sum(dist_params, axis=-1, keepdims=True)

        # logging
        per_predictor = tf.reduce_mean(pred_weights, axis=[1, 2])
        per_predictor_weights.append(per_predictor)

    if neptune_run:
        w_per_pred = tf.reduce_mean(per_predictor_weights, axis=[0]).numpy()
        for i_pred, w_pred in enumerate(w_per_pred):
            neptune_run[f'{env_name}/w_planning/pred_{i_pred}'].log(w_pred)
        #neptune_run[f'{env_name}/best_action_sequence'].log(top_a_sequence[0, :, 0].numpy())

    if debug_plot:
        fig.suptitle(f'Top {k} planning rollouts')
        ax0.set_ylabel('return')
        ax1.set_ylabel('trajectory length')
        ax0.set_xlabel('trajectory index')
        ax1.set_xlabel('trajectory index')
        plt.legend()
        plt.show()

    return top_a_sequence[0, :, 0], dist_params  # take best guess from last iteration and remove redundant dimension
    #return tfp.distributions.Categorical(probs=dist_params).sample(1)[..., tf.newaxis]


def plan_gaussian(predictor, vae, obs_history, act_history, n_actions, plan_steps, n_rollouts, n_iterations, top_perc,
                  gamma, env_name, neptune_run):
    """Crossentropy method, see algorithm 2.2 from https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf
    """
    # add axis for batch dim when encoding
    # add axis for batch dim when encoding
    preprocessed_start_samples = cast_and_normalize_images(obs_history.to_numpy())
    preprocessed_start_samples = vae.encode_to_indices(preprocessed_start_samples)
    preprocessed_start_actions = tf.cast(act_history.to_numpy(), tf.int32)
    # add axis for batch, then repeat n_rollouts times along batch dimension
    o_hist = tf.repeat(preprocessed_start_samples[tf.newaxis, ...], repeats=[n_rollouts], axis=0)
    a_hist = tf.repeat(preprocessed_start_actions[tf.newaxis, :, tf.newaxis], repeats=[n_rollouts], axis=0)
    mean = tf.random.uniform((plan_steps, 1), minval=0, maxval=n_actions - 1, dtype=tf.float32)
    scale = tf.random.uniform((plan_steps, 1), dtype=tf.float32)
    k = tf.cast(tf.round(n_rollouts * top_perc), tf.int32)

    for i_iter in range(n_iterations):
        # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
        # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
        a_new = tfp.distributions.MultivariateNormalDiag(loc=mean[:, 0], scale_diag=scale[:, 0]).sample(n_rollouts)
        a_new = tf.cast(tf.round(a_new), tf.int32)
        a_new = tf.clip_by_value(a_new, 0, n_actions - 1)
        a_new = tf.expand_dims(a_new, axis=-1)
        a_in = tf.concat([a_hist, a_new], axis=1)

        o_pred, r_pred, done_pred, pred_weights = predictor([o_hist, a_in])
        #r_pred = np.squeeze(r_pred.numpy())

        done_mask = tf.concat([tf.zeros((n_rollouts, 1), dtype=tf.float32), done_pred[:, :-1, 0]], axis=1)
        discount_factors = tf.map_fn(
            lambda d_traj: tf.scan(lambda cumulative, elem: cumulative * gamma * (1 - elem), d_traj, initializer=1.0),
            done_mask
        )
        discounted_returns = tf.reduce_sum(discount_factors * r_pred[:, :, 0], axis=1)

        top_returns, top_i_a_sequence = tf.math.top_k(discounted_returns, k=k)
        #top_a_sequence = tf.gather(a_new, top_i_a_sequence)[:, :, 0]
        top_a_sequence = tf.gather(a_new, top_i_a_sequence)
        #top_a_sequence = tf.cast(top_a_sequence, tf.float64)

        print(f'Top returns are: {top_returns}')
        #trajectory_video(cast_and_unnormalize_images(vae.decode_from_indices(o_pred[top_i_a_sequence[0], tf.newaxis, ...])), ['best sequence'])

        mean, scale = tf.nn.moments(tf.cast(top_a_sequence, tf.float32), axes=[0])

    print(f'Final mean: {mean}')
    print(f'Final var: {scale}')
    return top_a_sequence[0, :, 0]  # take best guess from last iteration and remove redundant dimension


def control(predictor, vae, env, env_info, env_name, plan_steps=50, warmup_steps=1, n_rollouts=64, n_iterations=5,
            top_perc=0.1, gamma=0.99, action_noise=0, consecutive_actions=1, max_steps=100, render=False, neptune_run=None):
    act_history = ValueHistory((), warmup_steps - 1)
    obs_history = ValueHistory(env_info['obs_shape'], warmup_steps)
    #available_actions = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    available_actions = []
    t = 0
    r = 0

    #last_observation = env.reset()
    #obs_history.append(last_observation)
    obs_history.append(env.reset())

    #for a in available_actions:
    #    last_observation, _, _, _ = env.step(a)
    #    act_history.append(a)
    #    obs_history.append(last_observation)
    #    env.render()
    #available_actions.clear()

    previous_dist_params = None
    while True:
        if render:
            env.render()

        # from tools import debug_visualize_observation_sequence
        # debug_visualize_observation_sequence(obs_history.to_numpy(), interval=250)
        # print(act_history.to_list())

        # propose new actions of none present
        if len(available_actions) == 0:
            preprocessed_start_samples = cast_and_normalize_images(obs_history.to_numpy())
            preprocessed_start_samples = vae.encode_to_indices(preprocessed_start_samples)
            preprocessed_start_actions = tf.cast(act_history.to_numpy(), tf.int32)
            actions, previous_dist_params = plan(predictor, previous_dist_params, preprocessed_start_samples,
                                                 preprocessed_start_actions, env_info['n_actions'],
                           plan_steps, n_rollouts, n_iterations, top_perc, gamma, action_noise, env_name, neptune_run)
            available_actions.extend([a for a in actions.numpy()[:consecutive_actions]])


        # pick first one and trash the rest if we do MPC
        action = available_actions.pop(0)
        #previous_dist_params = tf.concat([previous_dist_params[1:], tf.ones((1, env_info['n_actions'])) / env_info['n_actions']], axis=0)
        previous_dist_params = None

        act_names = ['up', 'right', 'down', 'left', 'noop']
        print(f'action: {act_names[action]}')
        observation, reward, done, info = env.step(action)

        # bookkeeping
        r += reward
        t += 1
        act_history.append(action)
        obs_history.append(observation)

        if done:
            print(f'Environment solved within {t} steps.')
            break
        #else:
        #    last_observation = observation
        if t == max_steps:
            print(f'FAILED to solve environment within {t} steps.')
            break

    env.close()

    return r, t