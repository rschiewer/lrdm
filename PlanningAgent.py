import tensorflow as tf
import copy
import tensorflow_probability as tfp
from predictors import *
from keras_vq_vae import *
from collections import namedtuple
import numpy as np
import numpy.lib.recfunctions as rfn
from tools import cast_and_normalize_images, cast_and_unnormalize_images, ValueHistory, prepare_predictor_data
from replay_memory_tools import line_up_observations


class ReplayMemory:

    def __init__(self, n_max_elements, s_obs, s_act=(), s_reward=(), custom_fields=None):
        self.n_max_elements = n_max_elements
        self.s_obs = s_obs
        self.s_act = s_act
        self.s_reward = s_reward

        data_dtype = [
            ('s', np.float32, s_obs),
            ('a', np.int32, s_act),
            ('r', np.float32, s_reward),
            ('done', np.int8, ()),
            ('i_tau', np.int32, ()),  # trajectory index
        ]

        if custom_fields is not None:
            assert isinstance(custom_fields, list)
            data_dtype.extend(custom_fields)
        else:
            custom_fields = []

        self._data_dtype = data_dtype
        self._return_data_dtype = copy.copy(data_dtype)
        self._return_data_dtype.insert(3, ('s_', np.float32, s_obs))

        self._custom_fields = custom_fields
        self._data = np.full(n_max_elements, -1, dtype=np.dtype(data_dtype))
        self._data_ptr = 0
        self._i_tau = 0
        self._n_added = 0
        self._rng = np.random.default_rng()
        self._transition_cache = []
        self._last_s_ = None

    @property
    def added_transitions(self):
        return self._n_added

    @property
    def valid_transitions(self):
        valid_trajectory = self._data['i_tau'] > -1
        terminal = self._data['done'] == True
        return np.count_nonzero(valid_trajectory) - np.count_nonzero(terminal)

    def add(self, s, a, r, s_, done, **custom_named_fields):
        """Adds a data tuple to the memory. """

        if self._custom_fields:
            additional_data = [custom_named_fields[field_dtype[0]] for field_dtype in self._custom_fields]
        else:
            additional_data = []

        if self._last_s_ is not None:
            if not np.all(self._last_s_ == s):
                raise ValueError('Last transition\'s s_ differs from current transition\'s s')

        self._transition_cache.append((s, a, r, done, self._i_tau, *additional_data))

        if done:
            self._transition_cache.append((s_, -1, -1, -1, self._i_tau, *[-1 for _ in self._custom_fields]))
            l_tau = len(self._transition_cache)

            if l_tau >= self.n_max_elements:
                raise RuntimeError(f'Current trajectory is longer than memory: {l_tau} vs. {self.n_max_elements}')

            # check if trajectory overflows memory
            if self._data_ptr + l_tau >= self.n_max_elements:
                self._data_ptr = 0

            # check if old trajectory gets partially overwritten
            last_i_tau = self._data[self._data_ptr + l_tau - 1]['i_tau']
            next_i_tau = self._data[self._data_ptr + l_tau]['i_tau']
            if last_i_tau == next_i_tau and last_i_tau != -1:
                self._data[self._data['i_tau'] == last_i_tau] = -1

            self._data[self._data_ptr: self._data_ptr + l_tau] = self._transition_cache

            self._data_ptr += l_tau
            self._transition_cache.clear()
            self._i_tau +=1
            self._n_added += l_tau - 1
            self._last_s_ = None
        else:
            self._last_s_ = s_

    def sample_transitions(self, n_transitions=1, debug=False):
        if self._n_added == 0:
            raise ValueError(f'Can\'t sample {n_transitions} samples, memory is empty!')

        i_valid = np.argwhere((self._data['i_tau'] > -1) & (self._data['done'] != -1)).flatten()
        if debug:
            i_samples = i_valid[0: n_transitions]
        else:
            i_samples = self._rng.choice(i_valid, n_transitions)
        return self._build_full_transitions(i_samples)

    def sample_trajectories(self, n_trajectories=1, n_steps=1, debug=False):
        if self._i_tau == 0:
            raise ValueError(f'Can\'t sample {n_trajectories} trajectories, memory is empty!')

        # find valid cells in the memory
        i_valid = np.argwhere(self._data['i_tau'] > -1)
        # find all available trajectory indices and the lengths of the respective trajectories
        i_tau, l_tau = np.unique(self._data[i_valid]['i_tau'], return_counts=True)
        # filter out too short trajectories
        valid_i_tau = i_tau[l_tau >= n_steps]
        valid_l_tau = l_tau[l_tau >= n_steps]

        # choose random indices
        if debug:
            i_tau_chosen, l_tau_chosen = valid_i_tau[0:n_trajectories], valid_l_tau[0:n_trajectories]
        else:
            i_tau_chosen, l_tau_chosen = self._rng.choice(zip(valid_i_tau, valid_l_tau), n_trajectories)

        batch = []
        for i, l in zip(i_tau_chosen, l_tau_chosen):
            if debug:
                subtraj_start = 0
            else:
                subtraj_start = self._rng.integers(0, l - n_steps)
            i_tau_current = np.argwhere(self._data['i_tau'] == i).flatten()
            batch.append(self._build_full_transitions(i_tau_current[subtraj_start: subtraj_start + n_steps]))

        return np.stack(batch)

    def _build_full_transitions(self, indices):
        incomplete_tuples = self._data[indices]
        complete_tuples = rfn.append_fields(incomplete_tuples, 's_', self._data[indices + 1]['s'], dtypes=np.float32)
        return complete_tuples

    def clear(self):
        self._data_ptr = 0
        self._n_added = 0
        self._i_tau = 0
        self._transition_cache.clear()
        self._last_s_ = None

    #def consistency_test(self):
    #    for i in range(min(self._n_added, self.n_max_transitions) - 1):
    #        assert (self._data[i]['s_'] == self._data[i + 1]['s']).all() or self._data[i]['done']


ParamsVae = namedtuple('ParamsVae', 'decay commitment_cost embedding_dim num_embeddings num_hiddens '
                                    'num_residual_hiddens num_residual_layers batch_size')
ParamsPredictor = namedtuple('ParamsPredictor', 'det_filters prob_filters n_models decider_lw batch_size '
                                                'n_timesteps n_warmup')
ParamsPlanning = namedtuple('ParamsPlanning', 'ctrl_n_plan_steps ctrl_n_warmup_steps ctrl_n_rollouts '
                                              'ctrl_n_iterations ctrl_top_perc ctrl_gamma ctrl_do_mpc '
                                              'ctrl_max_steps ctrl_render')

class SplitPlanAgent:

    def __init__(self, obs_shape, n_actions, mem_size, warmup_steps, params_vae, params_predictor, params_planning):
        self.vae = VectorQuantizerEMAKeras(train_data_variance=1,
                                           decay=params_vae.decay,
                                           commitment_cost=params_vae.commitment_cost,
                                           embedding_dim=params_vae.embedding_dim,
                                           num_embeddings=params_vae.num_embeddings,
                                           num_hiddens=params_vae.num_hiddens,
                                           num_residual_hiddens=params_vae.num_residual_hiddens,
                                           num_residual_layers=params_vae.num_residual_layers,
                                           grayscale_input=obs_shape[-1] == 1)
        self.vae.compile(optimizer=tf.optimizers.Adam())
        vae_index_matrix_shape = self.vae.compute_latent_shape(obs_shape)
        self.predictor = RecurrentPredictor(observation_shape=vae_index_matrix_shape,
                                            n_actions=n_actions,
                                            vqvae=self.vae,
                                            det_filters=params_predictor.det_filters,
                                            prob_filters=params_predictor.prob_filters,
                                            n_models=params_predictor.n_models,
                                            decider_filters=params_predictor.decider_lw)
        self.predictor.compile(optimizer=tf.optimizers.Adam())
        self.params_planning = params_planning
        self.params_vae = params_vae
        self.params_predictor = params_predictor

        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.warmup_steps = warmup_steps

        self.memory = ReplayMemory(mem_size, obs_shape)

    def plan(self, obs_history, act_history, n_actions, plan_steps, n_rollouts, n_iterations, top_perc, gamma):
        """Crossentropy method, see algorithm 2.2 from https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf,
        https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
        and https://towardsdatascience.com/cross-entropy-method-for-reinforcement-learning-2b6de2a4f3a0
        """

        # add axis for batch dim when encoding
        preprocessed_start_samples = cast_and_normalize_images(obs_history.to_numpy())
        preprocessed_start_samples = self.vae.encode_to_indices(preprocessed_start_samples)
        preprocessed_start_actions = tf.cast(act_history.to_numpy(), tf.int32)
        # add axis for batch, then repeat n_rollouts times along batch dimension
        o_hist = tf.repeat(preprocessed_start_samples[tf.newaxis, ...], repeats=[n_rollouts], axis=0)
        a_hist = tf.repeat(preprocessed_start_actions[tf.newaxis, :, tf.newaxis], repeats=[n_rollouts], axis=0)
        # initial params for sampling distribution
        dist_params = tf.ones((plan_steps, n_actions), dtype=tf.float32) / n_actions
        k = tf.cast(tf.round(n_rollouts * top_perc), tf.int32)

        assert n_iterations > 0, f'Number of iterations must be geater than 0 but is {n_iterations}'

        for i_iter in range(n_iterations):
            # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
            # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
            a_new = tfp.distributions.Categorical(probs=dist_params).sample(n_rollouts)[..., tf.newaxis]
            a_in = tf.concat([a_hist, a_new], axis=1)

            o_pred, r_pred, done_pred, pred_weights = self.predictor([o_hist, a_in])

            done_mask = tf.concat([tf.zeros((n_rollouts, 1), dtype=tf.float32), done_pred[:, :-1, 0]], axis=1)
            discount_factors = tf.map_fn(
                lambda d_traj: tf.scan(lambda cumulative, elem: cumulative * gamma * (1 - elem), d_traj,
                                       initializer=1.0),
                done_mask
            )

            discounted_returns = tf.reduce_sum(discount_factors * r_pred[:, :, 0], axis=1)
            top_returns, top_i_a_sequence = tf.math.top_k(discounted_returns, k=k)
            top_a_sequence = tf.gather(a_new, top_i_a_sequence)

            # MLE for categorical, see
            # https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
            # here we have multiple samples for MLE, which means the parameter update for one timestep is:
            # theta_i = sum_k a_ki / (sum_i sum_k a_ki) with i=action_index, k=sample
            top_a_sequence_onehot = tf.one_hot(top_a_sequence, n_actions, axis=-1)[:, :, 0, :]  # remove redundant dim
            numerator = tf.reduce_sum(top_a_sequence_onehot, axis=0)
            denominator = tf.reduce_sum(top_a_sequence_onehot, axis=[0, 2])[..., tf.newaxis]
            dist_params = numerator / denominator

        return top_a_sequence[0, :, 0]  # take best guess from last iteration and remove redundant dimension

    def train(self, env, steps):
        i_episode = 0
        i_step = 0
        while i_step < steps:
            act_history = ValueHistory((), self.params_planning.ctrl_n_warmup_steps - 1)
            obs_history = ValueHistory(self.obs_shape, self.params_planning.ctrl_n_warmup_steps)

            obs_history.append(env.reset())
            t_ep = 0
            r_ep = 0
            while True:
                if i_step < self.warmup_steps:
                    action = env.action_space.sample()
                else:
                    actions = self.plan(obs_history, act_history, self.n_actions, self.params_planning.ctrl_n_plan_steps,
                                        self.params_planning.ctrl_n_rollouts, self.params_planning.ctrl_n_iterations,
                                        self.params_planning.ctrl_top_perc, self.params_planning.ctrl_gamma)
                    action = actions[0].numpy()
                next_obs, reward, done, info = env.step(action)
                r_ep += reward

                if 'player_pos' in info.keys():
                    player_pos = info.pop('player_pos', None)
                elif env.agent_pos is not None:
                    player_pos = env.agent_pos
                else:
                    player_pos = [-43, -42]

                self.memory.add(obs_history[-1], action, reward, next_obs, done)
                if self.memory.added_transitions >= self.params_predictor.batch_size:
                    batch_predictor = self.memory.sample_trajectories(self.params_predictor.batch_size,
                                                                      self.params_predictor.n_timesteps,
                                                                      self.params_predictor.n_warmup)
                    self.train_step(batch_predictor)

                act_history.append(action)
                obs_history.append(next_obs)

                t_ep += 1
                i_step += 1

                if done:
                    print(f'\tEpisode {i_episode + 0} finished after {t_ep + 1} timesteps with reward {r_ep}')
                    break

                if i_step == steps:
                    print('Max number of interaction steps reached')
                    break

            i_episode += 1

        print('Training done')

    def train_step(self, trajectories):
        # vae training
        obs_only = line_up_observations(trajectories)
        obs_only = cast_and_normalize_images(obs_only)
        self.vae.data_variance = tf.math.reduce_variance(obs_only)
        self.vae.fit(obs_only)

        # predictor training
        enc_o, enc_o_, r, a, done = prepare_predictor_data(trajectories, self.vae, self.params_predictor.n_timesteps,
                                                           self.params_predictor.n_warmup_steps)
        self.predictor.fit([enc_o, a], [enc_o_, r, done])

    def train_step_wip(self, s, a, r, s_, done):
        norm_fact = np.prod(self.predictor.belief_state_shape)

        with tf.GradientTape() as tape:
            s_encoded = self.vae.encode_to_indices((tf.cast(s, tf.float32) / 255.0) - 0.5)
            s_next_encoded = self.vae.encode_to_indices((tf.cast(s_, tf.float32) / 255.0) - 0.5)
            s_one_hot = tf.one_hot(tf.cast(s_encoded, tf.int32), self.vae.num_embeddings, dtype=tf.float32)
            s_next_one_hot = tf.one_hot(tf.cast(s_next_encoded, tf.int32), self.vae.num_embeddings, dtype=tf.float32)
            s_pred, r_pred, done_pred, w_pred = self.predictor([s_one_hot, a], training=True)

            total_loss = 0.0
            total_obs_err = 0.0
            total_r_err = 0.0
            total_done_err = 0.0
            recon_loss = 0.0
            for i in range(self.predictor.n_models):
                curr_mdl_unweighted_obs_err = tf.losses.categorical_crossentropy(s_next_one_hot, s_pred[i])
                curr_mdl_obs_err = tf.reduce_mean(tf.reduce_mean(curr_mdl_unweighted_obs_err, axis=[2, 3]) * w_pred[i])
                curr_mdl_r_err = 0.1 * norm_fact * tf.reduce_mean(tf.losses.huber(r, r_pred[i]) * w_pred[i])
                curr_mdl_terminal_err = 0.01 * norm_fact * tf.reduce_mean(tf.losses.binary_crossentropy(done, done_pred[i]))

                total_obs_err += curr_mdl_obs_err
                total_r_err += curr_mdl_r_err
                total_done_err += curr_mdl_terminal_err
                total_loss += curr_mdl_obs_err + curr_mdl_r_err + curr_mdl_terminal_err


