import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from replay_memory_tools import cast_and_normalize_images


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
    r = 0
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

        reward += r
        t += 1

        if done:
            break
        else:
            last_observation = observation
    print(f'Environment solved within {t} steps.')
    env.close()

    return r, t