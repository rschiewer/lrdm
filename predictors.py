from tensorflow_probability import distributions as tfd
from tf_tools import *
from keras_vq_vae import VectorQuantizerEMAKeras


class AutoregressiveProbabilisticFullyConvolutionalMultiHeadPredictor(keras.Model):

    def __init__(self, observation_shape, n_actions, vqvae: VectorQuantizerEMAKeras,
                 det_filters=64, prob_filters=64, n_models=1, decider_lw=64,
                 open_loop_rollout_training=True, **kwargs):
        assert len(observation_shape) == 2, f'Expecting (w, h) shaped cb vector index matrices, got {len(observation_shape)}D'

        if kwargs.pop('debug_log', False):
            import datetime
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(f'./logs/{current_time}')
        else:
            self.summary_writer = None

        super(AutoregressiveProbabilisticFullyConvolutionalMultiHeadPredictor, self).__init__(**kwargs)

        self.s_obs = tuple(observation_shape)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self._train_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._vae_n_embeddings = vqvae.num_embeddings
        self._h_out_shape = (*observation_shape, det_filters)
        self._decider_lw = decider_lw
        self.open_loop_rollout_training = open_loop_rollout_training
        self.n_models = n_models
        self._vqvae = vqvae

        self.mdl_stack = []
        for i_mdl in range(n_models):
            det_model = self._det_state(det_filters, n_actions, observation_shape, vqvae)
            params_o_model = self._gen_params_o(self._h_out_shape, prob_filters, vqvae)
            params_r_model = self._gen_params_r(self._h_out_shape, prob_filters)
            self.mdl_stack.append((det_model, params_o_model, params_r_model))

        self.params_decider = self._gen_params_decider(self.s_obs, n_models, decider_lw, vqvae)

        self._obs_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='crossentropy_error')
        self._rew_accuracy = tf.keras.metrics.MeanSquaredError(name='mse')
        self._loss_tracker = tf.keras.metrics.Mean(name='loss')
        self._predictor_weights_tracker = [tf.keras.metrics.Mean(name=f'predictor_{i}_weight') for i in range(n_models)]

    def _gen_params_decider(self, s_obs, n_mdl, decider_lw, vqvae):
        def obs_flatten(inp):
            n_batch = tf.shape(inp)[0]
            n_time = tf.shape(inp)[1]
            return tf.reshape(inp, (n_batch, n_time, s_obs[0] * s_obs[1] * vqvae.embedding_dim))

        in_o = layers.Input((None, *s_obs, vqvae.num_embeddings), name='p_pred_o_in')
        lstm_c = layers.Input((decider_lw,), name='p_pred_lstm_c')
        lstm_h = layers.Input((decider_lw,), name='p_pred_lstm_h')
        vqvae_i_to_vec = self._gen_index_transform_layer(vqvae)

        x_params_pred = vqvae_i_to_vec(in_o)
        x_params_pred = layers.Lambda(lambda inp: obs_flatten(inp))(x_params_pred)
        x_params_pred = layers.Dense(64, activation='relu')(x_params_pred)
        x_params_pred = layers.LayerNormalization()(x_params_pred)
        x_params_pred = layers.Dense(64, activation='relu')(x_params_pred)
        x_params_pred = layers.LayerNormalization()(x_params_pred)
        x_params_pred, *lstm_states = layers.LSTM(decider_lw, return_state=True, return_sequences=True)(x_params_pred, initial_state=[lstm_c, lstm_h])
        x_params_pred = layers.Dense(n_mdl, activation=None, name='p_pred_out')(x_params_pred)

        return keras.Model(inputs=[in_o, lstm_c, lstm_h], outputs=[x_params_pred, lstm_states], name='p_pred_model')

    def _gen_params_r(self, h_out_shape, prob_filters):
        # stochastic model to implement p(r_t+1 | o_t, a_t, h_t)
        in_h = layers.Input(h_out_shape, name='p_r_in')
        x_params_r = layers.Flatten()(in_h)
        x_params_r = layers.Dense(prob_filters, activation='relu')(x_params_r)
        x_params_r = layers.LayerNormalization()(x_params_r)
        #x_params_r = layers.BatchNormalization()(x_params_r)
        x_params_r = layers.Dense(prob_filters, activation='relu')(x_params_r)
        x_params_r = layers.LayerNormalization()(x_params_r)
        #x_params_r = layers.BatchNormalization()(x_params_r)
        x_params_r = layers.Dense(2, activation=None, name='p_r_out')(x_params_r)

        return keras.Model(inputs=in_h, outputs=x_params_r, name='p_r_model')

    def _gen_params_o(self, h_out_shape, prob_filters, vqvae):
        # stochastic model to implement p(o_t+1 | o_t, a_t, h_t)
        in_h = layers.Input(h_out_shape, name='p_o_in')
        x_params_o = layers.Conv2D(prob_filters, kernel_size=5, padding='SAME', activation='relu')(in_h)
        x_params_o = layers.LayerNormalization()(x_params_o)
        #x_params_o = layers.BatchNormalization()(x_params_o)
        x_params_o = layers.Conv2D(prob_filters, kernel_size=3, padding='SAME', activation='relu')(x_params_o)
        x_params_o = layers.LayerNormalization()(x_params_o)
        #x_params_o = layers.BatchNormalization()(x_params_o)
        x_params_o = layers.Conv2D(vqvae.num_embeddings, kernel_size=3, padding='SAME', activation=None, name='p_o_out')(x_params_o)

        return keras.Model(inputs=in_h, outputs=x_params_o, name='p_o_model')

    def _det_state(self, det_filters, n_actions, s_obs, vqvae):
        # deterministic model to form state belief h_t = f(o_t-1, a_t-1, c_t-1)
        # note: h_t-1 is injected into the model not as explicit input but through previous LSTM states
        vqvae_i_to_vec = self._gen_index_transform_layer(vqvae)
        in_o = layers.Input((None, *s_obs, vqvae.num_embeddings), name='h_o_in')
        in_a = layers.Input((None, 1), name='h_a_in')
        lstm_c = layers.Input((*s_obs, det_filters), name='h_lstm_in0')
        lstm_h = layers.Input((*s_obs, det_filters), name='h_lstm_in1')

        o_cb_vectors = vqvae_i_to_vec(in_o)
        a_inflated = InflateActionLayer(s_obs, n_actions, True)(in_a)
        h = layers.Concatenate(axis=-1)([o_cb_vectors, a_inflated])
        h = layers.Conv2D(det_filters, kernel_size=5, padding='SAME', activation='relu')(h)
        h = layers.LayerNormalization()(h)
        #h = layers.BatchNormalization()(h)
        h = layers.Conv2D(det_filters, kernel_size=3, padding='SAME', activation='relu')(h)
        h = layers.LayerNormalization()(h)
        #h = layers.BatchNormalization()(h)
        h, *h_states = layers.ConvLSTM2D(det_filters, kernel_size=3, return_state=True,
                                                 return_sequences=True, padding='SAME',
                                                 name='h_out')(h, initial_state=[lstm_c, lstm_h])
        #h = layers.Conv2D(det_filters, kernel_size=3, padding='SAME', activation=None)(h)

        return keras.Model(inputs=[in_o, in_a, lstm_c, lstm_h], outputs=[h, h_states, a_inflated], name='h_model')

    def _gen_index_transform_layer(self, vqvae):
        def transform_fun(input):
            indices = tf.argmax(input, -1)
            index_matrices = vqvae.indices_to_embeddings(indices)
            return index_matrices

        return tf.keras.layers.Lambda(lambda inp: transform_fun(inp))

    def _temp_predictor_picker(self, training):
        if training:
            return 2
        else:
            return 0.01

    def _temp(self, training):
        if training:
            return 1
        else:
            return 0.01

    def _conv_lstm_start_states(self, n_batch, lstm_shape):
        return [tf.fill((n_batch, *lstm_shape), 0.0), tf.fill((n_batch, *lstm_shape), 0.0)]

    def _lstm_start_states(self, n_batch, lstm_lw):
        return [tf.fill((n_batch, lstm_lw), 0.0), tf.fill((n_batch, lstm_lw), 0.0)]

    def _o_dummy(self, n_batch):
        return tf.fill([n_batch, 1, *self.s_obs, self._vae_n_embeddings], 0.0)

    def _r_dummy(self, n_batch):
        return tf.fill([n_batch, 1, 1], 0.0)

    def _w_pred_dummy(self, n_batch):
        return tf.fill([n_batch, 1, len(self.mdl_stack)], 1.0 / len(self.mdl_stack))

    @tf.function
    def _rollout_closed_loop(self, inputs, training=None):
        o_in, a_in = inputs
        n_batch = tf.shape(a_in)[0]
        n_time = tf.shape(a_in)[1]

        n_warmup = tf.shape(o_in)[1]
        n_predict = tf.shape(a_in)[1]

        tf.debugging.assert_equal(n_warmup, n_predict, ('For closed loop rollout, observations and actions have to be '
                                                        f'provided in equal numbers, but are {n_warmup} and {n_predict}'))

        o_predictions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        r_predictions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i, (det_model, params_o_model, params_r_model) in enumerate(self.mdl_stack):
            dummy_states_h = self._conv_lstm_start_states(n_batch, self._h_out_shape)
            h, _hs, _ia = det_model([o_in, a_in] + dummy_states_h, training=training)
            h_flattened = tf.reshape(h, (-1, *self._h_out_shape))

            params_o = params_o_model(h_flattened, training=training)
            params_o = tf.reshape(params_o, (n_batch, n_time, *self.s_obs, self._vae_n_embeddings))
            params_r = params_r_model(h_flattened, training=training)
            params_r = tf.reshape(params_r, (n_batch, n_time, 2))

            o_pred = tfd.RelaxedOneHotCategorical(self._temp(training), params_o).sample()
            r_pred = tfd.Normal(loc=params_r[..., 0, tf.newaxis], scale=params_r[..., 1, tf.newaxis]).sample()
            #o_pred = tf.nn.softmax(params_o)
            #r_pred = params_r[:, :, 0, tf.newaxis]

            o_predictions = o_predictions.write(i, o_pred)
            r_predictions = r_predictions.write(i, r_pred)

        # retrospectively compute which predictor would have been chosen for which observation
        dummy_states_decider = self._lstm_start_states(n_batch, self._decider_lw)
        params_decider, _ = self.params_decider([o_in] + dummy_states_decider)
        #w_predictors = tfd.RelaxedOneHotCategorical(self._temp_predictor_picker(training), params_decider).sample()
        w_predictors = tf.nn.softmax(params_decider)
        w_predictors = tf.transpose(w_predictors, [2, 0, 1])  # bring predictor dimension to front

        return o_predictions.stack(), r_predictions.stack(), w_predictors

    @tf.function
    def rollout_open_loop(self, inputs, training=None):
        o_in, a_in = inputs

        n_batch = tf.shape(o_in)[0]
        n_warmup = tf.shape(o_in)[1]
        n_predict = tf.shape(a_in)[1]
        n_models = len(self.mdl_stack)
        t_start_feedback = n_warmup

        # pad o_in with zeros to avoid out of bounds indexing in _next_input (if-else tf autograph bullshit)
        o_in_padded = o_in #tf.concat([o_in, tf.zeros((n_batch, n_predict - n_warmup, *self.s_obs, self._vae_n_embeddings))],
                           #     axis=1)

        tf.debugging.assert_less(n_warmup, n_predict, ('For rollout, less observations than actions are expected, '
                                                       f'but I got {n_warmup} observation and {n_predict} action '
                                                       f'steps.'))

        # store for rollout results
        o_predictions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        r_predictions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        w_predictors = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # placeholders for start
        states_h = tf.stack([self._conv_lstm_start_states(n_batch, self._h_out_shape) for _ in range(n_models)])
        states_decider = self._lstm_start_states(n_batch, self._decider_lw)
        o_pred = tf.stack([self._o_dummy(n_batch) for _ in range(n_models)])
        r_pred = tf.stack([self._r_dummy(n_batch) for _ in range(n_models)])
        w_pred = self._w_pred_dummy(n_batch)  # same probability for each model before first observation

        # rollout start
        for i_t in range(n_predict):
            #tf.autograph.experimental.set_loop_options(shape_invariants=[(w_pred, tf.TensorShape([None, None, n_models]))])

            o_step = tf.TensorArray(tf.float32, size=n_models)
            r_step = tf.TensorArray(tf.float32, size=n_models)
            states_h_step = tf.TensorArray(tf.float32, size=n_models)

            # choose next current observation based on predictor weights of last iteration (i.e. given last observation)
            o_next, a_next = self._next_input(o_in_padded, a_in, o_pred, w_pred, i_t, t_start_feedback)

            # pick a predictor given current observation
            params_decider, states_decider = self.params_decider([o_next] + states_decider)
            params_decider = params_decider[:, 0, tf.newaxis, :]  # make tensor shape explicit (None, 1, n_models) for autograph
            #w_pred = tfd.RelaxedOneHotCategorical(self._temp_predictor_picker(training), params_decider).sample()
            w_pred = tf.nn.softmax(params_decider)

            # do predictions with all predictors
            for i_m, (h_mdl, params_o_mdl, params_r_mdl) in enumerate(self.mdl_stack):
                o_pred, r_pred, model_h_state = self._open_loop_step(h_mdl, params_o_mdl, params_r_mdl, o_next, a_next, states_h[i_m], training)
                o_step = o_step.write(i_m, o_pred)
                r_step = r_step.write(i_m, r_pred)
                states_h_step = states_h_step.write(i_m, model_h_state)

            o_pred = o_step.stack()
            r_pred = r_step.stack()
            states_h = states_h_step.stack()

            o_predictions = o_predictions.write(i_t, o_pred)
            r_predictions = r_predictions.write(i_t, r_pred)
            w_predictors = w_predictors.write(i_t, w_pred[:, 0])

        # currently, outputs have two time dimensions (timestep, predictor, batch, 1 (timestep), width, height, one_hot_vec)
        o_predictions = o_predictions.stack()[:, :, :, 0, ...]
        r_predictions = r_predictions.stack()[:, :, :, 0, ...]
        w_predictors = w_predictors.stack()
        # currently, outputs are ordered like (timestep, predictor, batch, width, height, one_hot_vec)
        # they need to be transposed to (predictor, batch, timestep, width, height, one_hot_vec)
        o_predictions = tf.transpose(o_predictions, [1, 2, 0, 3, 4, 5])
        r_predictions = tf.transpose(r_predictions, [1, 2, 0, 3])
        w_predictors = tf.transpose(w_predictors, [2, 1, 0])

        return o_predictions, r_predictions, w_predictors

    @tf.function
    def _open_loop_step(self, det_model, params_o_model, params_r_model, o_inp, a_inp, states_h, training):
        n_batch = tf.shape(a_inp)[0]

        h, states_h, _ia = det_model([o_inp, a_inp, states_h[0], states_h[1]], training=training)
        h_flattened = tf.reshape(h, (-1, *self._h_out_shape))
        params_o = params_o_model(h_flattened, training=training)
        params_o = tf.reshape(params_o, (n_batch, 1, *self.s_obs, self._vae_n_embeddings))
        params_r = params_r_model(h_flattened, training=training)
        params_r = tf.reshape(params_r, (n_batch, 1, 2))

        o_pred = tfd.RelaxedOneHotCategorical(self._temp(training), params_o).sample()
        r_pred = tfd.Normal(loc=params_r[..., 0, tf.newaxis], scale=params_r[..., 1, tf.newaxis]).sample()

        return o_pred, r_pred, states_h

    @tf.function
    def call(self, inputs, mask=None, training=None):
        o_in, a_in = inputs

        # convert observations to one_hot
        one_hot_obs = tf.one_hot(tf.cast(o_in, tf.int32), self._vae_n_embeddings, axis=-1)
        inputs = (one_hot_obs, a_in)

        if self.open_loop_rollout_training:
            trajectories = self.rollout_open_loop(inputs, training)
        else:
            trajectories = self._rollout_closed_loop(inputs, training)

        o_predicted, r_predicted, w_predictors = trajectories

        if not training:
            o_predicted, r_predicted = self.most_probable_trajectories(o_predicted, r_predicted, w_predictors)
            o_predicted = tf.argmax(o_predicted, axis=-1)

        return o_predicted, r_predicted, w_predictors

    @tf.function
    def most_probable_trajectories(self, o_predictions, r_predictions, w_predictors):
        # o_predictions shape: (predictor, batch, timestep, width, height, one_hot_index)
        # push predictor index backwards for easier selection
        o_predictions = tf.transpose(o_predictions, [1, 2, 0, 3, 4, 5])
        r_predictions = tf.transpose(r_predictions, [1, 2, 0, 3])
        w_predictors = tf.transpose(w_predictors, [1, 2, 0])
        # select only the most probable predictor per step
        i_predictor = tf.expand_dims(tf.argmax(w_predictors, axis=-1), axis=-1)
        o_predictions = tf.gather_nd(o_predictions, i_predictor, batch_dims=2)
        r_predictions = tf.gather_nd(r_predictions, i_predictor, batch_dims=2)
        return o_predictions, r_predictions

    @tf.function
    def train_step(self, data):
        tf.assert_equal(len(data), 2), f'Need tuple (x, y) for training, got {len(data)}'

        x, y = data

        # lock parameters of vqvae to make sure they are not updated
        vqvae_is_trainable = self._vqvae.trainable
        self._vqvae.trainable = False

        with tf.GradientTape() as tape:
            o_groundtruth = tf.one_hot(tf.cast(y[0], tf.int32), self._vae_n_embeddings, dtype=tf.float32)
            r_groundtruth = y[1]

            #if self.summary_writer:
            #    with self.summary_writer.as_default():
            #        tf.summary.trace_on(graph=True, profiler=True)
            #        o_predictions, r_predictions, w_predictors = self(x, training=True)
            #        tf.summary.trace_export(name='Convolutional_Predictor_Trace', step=self._train_step.value(), profiler_outdir='graph')
            #else:
            o_predictions, r_predictions, w_predictors = self(x, training=True)

            total_loss = 0.0
            # this might be wrong, in every timestep only the chosen predictor should be updated
            # but currently, there are all predictors updated weighted with the probability that they are chosen
            for i in range(self.n_models):
                o_pred = o_predictions[i]
                r_pred = r_predictions[i]
                w_predictor = w_predictors[i]
                curr_mdl_obs_err = tf.reduce_sum(tf.losses.categorical_crossentropy(o_groundtruth, o_pred), axis=[2, 3]) * w_predictor
                curr_mdl_r_err = tf.losses.mean_squared_error(r_groundtruth, r_pred) * w_predictor
                total_loss += tf.reduce_mean(curr_mdl_obs_err) + tf.reduce_mean(curr_mdl_r_err)
                #total_loss += 0.1 * tf.reduce_sum(tf.math.multiply(w_predictor, tf.math.log(w_predictor)))  # regularization to incentivize picker to not let a predictor starve

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_weights)
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self._vqvae.trainable = vqvae_is_trainable

        o_most_probable, r_most_probable = self.most_probable_trajectories(o_predictions, r_predictions, w_predictors)

        self._obs_accuracy.update_state(o_groundtruth, o_most_probable)
        self._rew_accuracy.update_state(r_groundtruth, r_most_probable)
        self._loss_tracker.update_state(total_loss)

        self._train_step.assign(self._train_step.value() + 1)

        #weight_stats = {f'w{i}': tf.reduce_mean(w_predictors, axis=[1, 2])[i] for i in range(tf.shape(w_predictors)[0])}
        return {'loss': self._loss_tracker.result(),
                'most_probable_observation_error': self._obs_accuracy.result(),
                'most_probable_r_error': self._rew_accuracy.result(),
                't': self._temp(True),
                'w0': tf.reduce_mean(w_predictors, axis=[1, 2])[0],
                #'w1': tf.reduce_mean(w_predictors, axis=[1, 2])[1],
                #'w2': tf.reduce_mean(w_predictors, axis=[1, 2])[2]
                }#.update(weight_stats)

    @tf.function
    def _next_input(self, o_in, a_in, o_last, w_pred, i_t, feedback):
        if i_t < feedback:
            o_next = o_in[:, i_t, tf.newaxis]
        else:
            best_fit = tf.expand_dims(tf.argmax(w_pred, axis=-1), -1)
            # (predictor, batch, time, w, h, c) -> (batch, time, predictor, w, h, c)
            o_next = tf.transpose(o_last, [1, 2, 0, 3, 4, 5])
            # select most probable prediction per batch element and timestep
            o_next = tf.gather_nd(o_next, best_fit, batch_dims=2)

        return o_next, a_in[:, i_t, tf.newaxis]