import tensorflow as tf
import copy
import tensorflow_probability as tfp
from predictors import *
from keras_vq_vae import *
from collections import namedtuple
import numpy as np
import numpy.lib.recfunctions as rfn


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
            if not self._last_s_ == s:
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

    def sample_trajectories(self, n_trajectories=1, n_steps=1, debug=True):
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
ParamsPredictor = namedtuple('ParamsPredictor', 'det_filters prob_filters n_models decider_lw lr batch_size')


class SplitPlanAgent:

    def __init__(self, obs_shape, n_actions, mem_size, params_vae, params_predictor):
        self.vae = VectorQuantizerEMAKeras(train_data_variance=1,
                                           decay=params_vae.decay,
                                           commitment_cost=params_vae.commitment_cost,
                                           embedding_dim=params_vae.embedding_dim,
                                           num_embeddings=params_vae.num_embeddings,
                                           num_hiddens=params_vae.num_hiddens,
                                           num_residual_hiddens=params_vae.num_residual_hiddens,
                                           num_residual_layers=params_vae.num_residual_layers,
                                           grayscale_input=obs_shape[-1] == 1)
        self.vae.compile(optimizer=tf.optimizers.Adam(params_vae.lr))
        self.predictor = RecurrentPredictor(observation_shape=obs_shape,
                                            n_actions=n_actions,
                                            vqvae=self.vae,
                                            det_filters=params_predictor.det_filters,
                                            prob_filters=params_predictor.prob_filters,
                                            n_models=params_predictor.n_models,
                                            decider_lw=params_predictor.decider_lw)
        self.predictor.compile(optimizer=tf.optimizers.Adam(params_predictor.lr))

        self.obs_shape = obs_shape
        self.n_actions = n_actions

    def plan(self, start_sample, plan_steps, n_rollouts, n_iterations=10, top_perc=0.1):
        """Crossentropy method, see algorithm 2.2 from https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf."""

        # add axis for batch dim when encoding
        encoded_start_sample = self.vae.encode_to_indices(start_sample[tf.newaxis, ...])
        # add axis for time, then repeat n_rollouts times along batch dimension
        o_in = tf.repeat(encoded_start_sample[tf.newaxis, ...], repeats=[n_rollouts], axis=0)
        dist_params = tf.zeros((1, plan_steps, self.n_actions), dtype=tf.float32)
        k = tf.cast(tf.round(n_rollouts * top_perc), tf.int32)

        for i_iter in range(n_iterations):
            # generate one action vector per rollout trajectory (we generate n_rollouts trajectories)
            # each timestep has the same parameters for all rollouts (so we need plan_steps * n_actions parameters)
            a_in = tfp.distributions.Categorical(logits=dist_params).sample(n_rollouts)

            o_pred, r_pred, pred_weights = self.predictor([o_in, a_in])
            top_returns, top_i_a = tf.math.top_k(tf.reduce_sum(r_pred, axis=1), k=k)
            top_a = tf.gather(a_in, top_i_a)

            # MLE for categorical, see
            # https://math.stackexchange.com/questions/2725539/maximum-likelihood-estimator-of-categorical-distribution
            # here we have multiple samples for MLE, which means the parameter update for one timestep is:
            # theta_i = sum_k a_ki / (sum_i sum_k a_ki) with i=action_index, k=sample
            dist_params = tf.reduce_sum(top_a, axis=1) / tf.reduce_sum(top_a, axis=[0, 1])

        return tfp.distributions.Categorical(logits=dist_params).sample()

    def train_env(self, env, steps):
        i_episode = 0
        for i_step in range(steps):
            pass






