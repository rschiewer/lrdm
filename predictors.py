import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import copy

from tensorflow import keras
from tensorflow.keras import layers
from collections.abc import Iterable
from functools import partial
from tf_tools import *
from keras_vq_vae import VectorQuantizerEMAKeras


class OldPredictor(keras.Model):

    def __init__(self, input_sizes, output_sizes, timesteps=None, return_sequences=False, pre_lws=32,
                 intermediate_lws=64, post_lws=32, lstm_width=0, **kwargs):
        super(OldPredictor, self).__init__(**kwargs)

        assert (
                           return_sequences is True and lstm_width > 0) or return_sequences is False, 'return_sequences works with LSTM only'

        # input checks
        if type(input_sizes) is int:
            input_sizes = [input_sizes]
        assert isinstance(input_sizes, Iterable), 'input_sizes must be int or an iterable containing ints'

        for in_size in input_sizes:
            assert type(in_size) is int, 'input_sizes can be of int type only'

        if type(output_sizes) is int:
            output_sizes = [output_sizes]
        assert isinstance(output_sizes, Iterable), 'onput_sizes must be int or an iterable containing ints'

        if type(pre_lws) is int:
            pre_lws = [[pre_lws] for inp in input_sizes]
        elif isinstance(pre_lws, Iterable):
            assert len(pre_lws) == len(
                input_sizes), 'If pre_lws is iterable, it must contain exactly one iterable per input'
            assert all(
                [isinstance(elem, Iterable) for elem in pre_lws]), 'If pre_lws is iterable, it must contain iterables'
        assert isinstance(pre_lws,
                          Iterable), 'pre_lws must be int, an iterable containing ints or an iterable of iterables containint ints'

        if type(intermediate_lws) is int:
            intermediate_lws = [intermediate_lws]
        assert isinstance(intermediate_lws, Iterable), 'intermediate_lws must be int or an iterable containing ints'

        if type(post_lws) is int:
            post_lws = [post_lws]
        assert isinstance(post_lws, Iterable), 'post_lws must be int or an iterable containing ints'

        # build input layers
        if lstm_width:
            # prepend placeholder for time dimension
            model_inputs = [layers.Input(shape=(timesteps, in_size)) for in_size in input_sizes]
        else:
            assert timesteps is not None, 'If no LSTM layer is used, a fixed time dimension for the input is required'
            model_inputs = [layers.Input(shape=(timesteps, in_size)) for in_size in input_sizes]

        # build each of the individual input branches
        x = []
        for idx, branch in enumerate(pre_lws):
            x.append(model_inputs[idx])
            for lw in branch:
                x[idx] = layers.Dense(lw, activation='relu')(x[idx])

        # if there are multiple inputs, concatenate; else remove list
        if len(x) > 1:
            x = layers.Concatenate()(x)
        else:
            x = x[0]

        # build pre-lstm, lstm and post-lstm layers
        for lw in intermediate_lws:
            x = layers.Dense(lw, activation='relu')(x)

        if lstm_width:
            init_h = layers.Input(shape=(lstm_width,), name='init_h')
            init_c = layers.Input(shape=(lstm_width,), name='init_c')
            x, *lstm_states = layers.LSTM(lstm_width,
                                          #kernel_regularizer=tf.keras.regularizers.L1(0.001),
                                          #activity_regularizer=tf.keras.regularizers.L2(0.001),
                                          #dropout=0.1,
                                          return_sequences=return_sequences,
                                          return_state=True)(x, initial_state=[init_h, init_c])
            model_inputs.append(init_h)
            model_inputs.append(init_c)

        for layer in post_lws:
            x = layers.Dense(lw, activation='relu')(x)

        # apply output layers
        train_outputs = [layers.Dense(lw, activation=None)(x) for lw in output_sizes]
        predict_outputs = copy.copy(train_outputs)

        if lstm_width:
            predict_outputs.append(lstm_states[0])
            predict_outputs.append(lstm_states[1])
        else:
            # squeeze output of predict model, since this could be list of length one
            if len(predict_outputs) == 1:
                predict_outputs = predict_outputs[0]

        # squeeze output of training model, since this could be list of length one
        if len(train_outputs) == 1:
            train_outputs = train_outputs[0]

        self.train_mdl = keras.Model(inputs=model_inputs, outputs=train_outputs, name='train_model')
        self.train_mdl.compile(optimizer=tf.optimizers.Adam(), loss='mse')
        self.pred_mdl = keras.Model(inputs=model_inputs, outputs=predict_outputs, name='predict_model')

        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.lstm_width = lstm_width

        build_shape = [tf.TensorShape(m_inp.shape) for m_inp in model_inputs]
        self.build(build_shape)

    def gen_init_lstm_states(self, batch_size):
        lstm_h = np.zeros(shape=(batch_size, self.lstm_width), dtype=np.float32)
        lstm_c = np.zeros(shape=(batch_size, self.lstm_width), dtype=np.float32)

        return lstm_h, lstm_c

    def summary(self):
        super(OldPredictor, self).summary()
        self.train_mdl.summary()
        self.pred_mdl.summary()

    @tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None):
        if training:
            return self.train_mdl(inputs, training=training)
        else:
            return self.pred_mdl(inputs, training=training)

    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self.train_mdl(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            for l in self.losses:
                loss += l

        # Compute gradients
        gradients = tape.gradient(loss, self.train_mdl.trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.train_mdl.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def _batch_dim(self, inp):
        if np.ndim(inp) >= 2:
            return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    def _time_dim(self, inp):
        if np.ndim(inp) == 3:
            return np.shape(inp)[1]
        else:
            raise ValueError('1D or 2D array found, can\'t infer time dimension')

    def _data_dim(self, inp):
        return np.shape(inp)[-1]

    def _has_time_dim(self, inp):
        if np.ndim(inp) == 3:
            return True
        else:
            return False

    def _rollout_input_checks(self, input_data):
        # input format
        assert isinstance(input_data, list) or np.ndim(input_data) == 4, 'Need list of inputs or 4D numpy array'

        # enough inputs
        assert len(input_data) == len(self.input_sizes), 'Need at least data for one timestep per input'

        # batch size
        assert all([self._batch_dim(inp) == len(input_data[0]) for inp in input_data]), ('Inconsistent '
                                                                                         'batch size')
        # check for time dimension
        assert all([np.ndim(inp) == 3 for inp in input_data]), ('All input must have shape (batch_dim, ',
                                                                'time_dim, input_dim)')
        # check input shapes
        assert all([self._data_dim(inp) == in_shp
                    for inp, in_shp in zip(input_data, self.input_sizes)]), ('Shape mismatch in input')

    def _update_feedback_dict(self, input_data, num_steps, feedback):
        feedback = feedback or {}

        i_out = iter(range(len(self.pred_mdl.outputs)))
        for i_in, inp in enumerate(input_data):
            if i_in in feedback:
                continue
            if self._time_dim(inp) < num_steps:
                try:
                    feedback[i_in] = next(i_out)
                except StopIteration as ex:
                    raise ValueError('Out of output slots for feedback, need more input data timesteps') from ex
            else:
                feedback[i_in] = None

        return feedback

    def _gather_input(self, input_data, outp_last, feedback, i_t):
        inp_curr = []

        for i_in, inp in enumerate(input_data):
            if self._has_time_dim(inp) and i_t < self._time_dim(inp):
                inp_curr.append(inp[:, i_t, np.newaxis])  # add back time dimension
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])  # has same dimensions as input, so time dimension is present

        # inject lstm states
        if self.lstm_width:
            if i_t == 0:
                batch_size = self._batch_dim(input_data[0])
                init_states = self.gen_init_lstm_states(batch_size)
                inp_curr.extend(list(init_states))
            else:
                inp_curr.extend(outp_last[-2:])

        return inp_curr

    def _store_output(self, outp_last, traj_history, i_t):
        for outp, save_slot in zip(outp_last, traj_history):
            if self._has_time_dim(outp):
                outp = outp[:, 0]
            save_slot[:, i_t] = outp

    def rollout(self, input_data, num_steps, feedback=None, input_checks=True):
        if isinstance(input_data, np.ndarray):
            if np.ndim(input_data) == 3:
                input_data = [input_data]

        if input_checks:
            self._rollout_input_checks(input_data)

        batch_size = self._batch_dim(input_data[0])
        feedback = self._update_feedback_dict(input_data, num_steps, feedback)
        traj_history = [np.zeros(shape=(batch_size, num_steps, self._data_dim(outp)))
                        for outp in self.pred_mdl.outputs]

        outp_last = None
        init_h, init_c = self.gen_init_lstm_states(batch_size)
        inp_curr = [input_data[0][:, 0:1], init_h, init_c]
        for i_t in range(num_steps):
            inp_curr_tmp = self._gather_input(input_data, outp_last, feedback, i_t)
            outp_last = self.pred_mdl.predict(inp_curr_tmp)
            print([(ic - ic_tmp).mean() for ic, ic_tmp in zip(inp_curr, inp_curr_tmp)])
            self._store_output(outp_last, traj_history, i_t)
            inp_curr = outp_last

        #print(input_data)
        #print(np.array([inp[0] for inp in inp_history]).squeeze())
        #print('----')

        return traj_history


class Predictor(keras.Model):

    def __init__(self, input_sizes, output_sizes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lws=32, **kwargs):
        super(Predictor, self).__init__(**kwargs)

        # input checks
        input_sizes = self._to_list(input_sizes)
        output_sizes = self._to_list(output_sizes)
        if type(pre_lws) is int:
            pre_lws = tuple([(pre_lws,) for inp in input_sizes])
        intermediate_lws = self._to_list(intermediate_lws)
        lstm_lws = self._to_list(lstm_lws)
        post_lws = self._to_list(post_lws)

        in_branches = []
        for branch_lws in pre_lws:
            in_branches.append([layers.Dense(lw, activation='relu') for lw in branch_lws])
        self.in_branches = self._to_tuple(in_branches)

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate()
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in intermediate_lws])
        self.lstm_layers = self._to_tuple([layers.LSTM(lw, return_sequences=True, return_state=True) for lw in lstm_lws])
        self.post_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in post_lws])
        self.model_outputs = self._to_tuple([layers.Dense(lw, activation=None) for lw in output_sizes])

        self.input_sizes = self._to_tuple(input_sizes)
        self.output_sizes = self._to_tuple(output_sizes)
        self.lstm_width = self._to_tuple(lstm_lws)
        self._warmup_steps = 1
        self._predict_steps = 1
        self._feedback = {i: -1 for i in range(len(input_sizes))}

        # build model
        # batch size=32, timesteps=5
        #self._predict_steps = 5
        #self._warmup_steps = 2
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        #self(sample_data)

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    @property
    def warmup_steps(self):
        return self._warmup_steps

    @warmup_steps.setter
    def warmup_steps(self, new_value):
        self._warmup_steps = new_value

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_value):
        self._predict_steps = new_value

    @property
    def feedback(self):
        return self._feedback

    @feedback.setter
    def feedback(self, new_value):
        self._feedback = new_value

    def _eval_lstm_init(self, data):
        #states_mem = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        states_mem = []
        for l in self.lstm_layers:
            data, *s = l(data)
            #tf.print('before concat: ', tf.shape(s))
            #s = tf.concat(s, axis=0)
            #tf.print('after concat: ', tf.shape(s))
            #states_mem = states_mem.write(states_mem.size(), s)
            states_mem.append(s)
        #return data[:, -1, tf.newaxis], tuple(states_mem)
        #states_mem = states_mem.stack()
        return data[:, -1, np.newaxis], states_mem

    def _eval_lstm(self, data, init_states):
        states_mem = []
        for l, init_l in zip(self.lstm_layers, init_states):
            data, *s = l(data, init_l)
            states_mem.append(s)
        return data, states_mem
        #states_mem = []
        #for l, init_s in zip(self.lstm_layers, init_states):
        #    data, *s = l(data, initial_state=init_s)
        #    states_mem.append(s)
        #return data, tuple(states_mem)

    def evaluate_step(self, inputs, states):
        # wrap to list in case there is only one input stream
        #if type(inputs) is not tuple:
        #    inputs = (inputs, )

        # input branches
        #for inp, in_branch in zip(inputs, self.in_branches):
        #    x = inp
        #    for layer in in_branch:
        #        x = layer(x)
        #    data.append(x)
        # concatenate
        #data = self.concat(data)

        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layers
        if states is None:
            data, states = self._eval_lstm_init(data)
        else:
            data, states = self._eval_lstm(data, states)
        # post lstm layers
        for layer in self.post_layers:
            data = layer(data)
        # output layers
        outputs = tuple([layer(data) for layer in self.model_outputs])
        #outputs = tf.TensorArray(tf.float32, size=len(self.output_sizes))
        #for i_out in range(len(self.model_outputs)):
        #    outputs.write(i_out, self.model_outputs[i_out](data))
        #outputs = outputs.stack()
        #outputs = tf.ragged.stack([l(data) for l in self.model_outputs])

        return outputs, states

    def _detect_feedback(self, feedback, inputs, warmup_steps, predict_steps):
        i_out = 0
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < warmup_steps + predict_steps - 1:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(self.model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {warmup_steps + predict_steps - 1}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_sizes[i_in] != self.output_sizes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match:'))

        return feedback_out

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        if type(inputs) is tuple and len(inputs) == 3:
            inputs = (inputs, )
        elif len(inputs) == 4 and type(inputs) is not tuple:
            inputs = tuple(inputs)

        n_warmup = self._warmup_steps
        n_predict = self._predict_steps
        feedback = self._detect_feedback(self._feedback, inputs, n_warmup, n_predict)

        # generate one list per output
        #output_buffer = tf.nest.map_structure(lambda _: tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True), self.model_outputs)
        output_buffer = [[] for _ in self.model_outputs]

        # use only the first self.warmup_steps number of steps for warmup
        # this yields one prediction step
        #warmup_inputs = tf.nest.map_structure(lambda inp: inp[:, 0:n_warmup], inputs)
        warmup_inputs = [inp[:, 0:n_warmup] for inp in inputs]
        #warmup_inputs = tf.ragged.stack([inp[:, 0:n_warmup] for inp in inputs], axis=0)

        last_prediction, states = self.evaluate_step(warmup_inputs, None)
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

        for i_t in range(n_predict - 1):  # warmup yields first prediction, so do one less
            next_input = self._gather_next_input(inputs, last_prediction, i_t + n_warmup, feedback)

            last_prediction, states = self.evaluate_step(next_input, states)
            for i_out, out_buf in enumerate(output_buffer):
                out_buf.append(last_prediction[i_out])

        #output_buffer = tf.nest.map_structure(lambda outp: outp.stack(), output_buffer)
        #output_buffer = tf.nest.map_structure(lambda outp: tf.transpose(outp, [1, 0, 2]), output_buffer)
        #output_buffer = [output_buffer[i] for i in range(output_buffer.shape[0])]
        output_buffer = [tf.concat(outp, axis=1) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    """
    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)
        tf.print([tf.shape(layer) for layer in gradients])
        tf.print(tf.reduce_mean(gradients[0], axis=0))
        tf.print(tf.reduce_mean(gradients[5], axis=0))
        tf.print(tf.reduce_mean(gradients[4], axis=0))
        #tf.print(tf.shape(gradients[4]))
        #tf.print(tf.shape(gradients[5]))
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    """

    @staticmethod
    def _batch_dim(inp):
        if len(np.shape(inp)) >= 2:
            return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) == 3:
            return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        return np.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if np.ndim(inp) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if i_t < self._time_dim(inp):
                inp_curr.append(inp[:, i_t, np.newaxis])  # add back time dimension
                #print(f'timestep {i_t} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])  # has same dimensions as input, so time dimension is present
                #print(f'timestep {i_t} input{i_in}: using feedback')

        return inp_curr


class AutoregressivePredictor_old(keras.Model):

    def __init__(self, input_sizes, output_sizes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lws=32, **kwargs):
        super(AutoregressivePredictor_old, self).__init__(**kwargs)

        # input checks
        input_sizes = self._to_list(input_sizes)
        output_sizes = self._to_list(output_sizes)
        if type(pre_lws) is int:
            pre_lws = tuple([(pre_lws,) for inp in input_sizes])
        intermediate_lws = self._to_list(intermediate_lws)
        lstm_lws = self._to_list(lstm_lws)
        post_lws = self._to_list(post_lws)

        #elif isinstance(pre_lws, Iterable):
        #    assert len(pre_lws) == len(input_sizes), 'If pre_lws is iterable, it must contain exactly one iterable per input'
        #    assert all([isinstance(elem, Iterable) for elem in pre_lws]), 'If pre_lws is iterable, it must contain iterables'
        #assert isinstance(pre_lws, Iterable), 'pre_lws must be int, an iterable containing ints or an iterable of iterables containint ints'

        #if type(intermediate_lws) is int:
        #    intermediate_lws = [intermediate_lws]
        #assert isinstance(intermediate_lws, Iterable), 'intermediate_lws must be int or an iterable containing ints'

        #if type(lstm_lws) is int:
        #    lstm_lws = [lstm_lws]
        #assert isinstance(lstm_lws, Iterable), 'lstm_lws must be int or an iterable containing ints'

        #if type(post_lws) is int:
        #    post_lws = [post_lws]
        #assert isinstance(post_lws, Iterable), 'post_lws must be int or an iterable containing ints'

        # build each of the individual input branches
        #self.in_branches = []
        #for branch_lws, in_size in zip(pre_lws, input_sizes):
            #branch_layers = [layers.Dense(branch_lws[0], activation='relu', input_shape=(None, None, in_size))]
            #for lw in branch_lws[1:]:
            #    branch_layers.append(layers.Dense(lw, activation='relu'))
            #self.in_branches.append(tuple(branch_layers))
        #self.in_branches = tuple(self.in_branches)

        in_branches = []
        for branch_lws in pre_lws:
            in_branches.append([layers.TimeDistributed(layers.Dense(lw, activation='relu')) for lw in branch_lws])
        self.in_branches = self._to_tuple(in_branches)

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate(axis=-1)
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = self._to_tuple([layers.TimeDistributed(layers.Dense(lw, activation='relu')) for lw in intermediate_lws])
        self.lstm_layers = self._to_tuple([layers.GRU(lw, return_sequences=True, return_state=True) for lw in lstm_lws])
        #self.lstm_layers = [layers.LSTM(lw, return_sequences=True, return_state=True) for lw in lstm_lws[0:-1]]
        #self.lstm_layers.append(layers.LSTM(lstm_lws[-1], return_sequences=False, return_state=True))
        #self.lstm_layer = layers.LSTM(lstm_lws, return_sequences=False, return_state=True)
        #self.lstm_cell = layers.LSTMCell(lstm_width)
        #self.lstm_layer = layers.RNN(self.lstm_cell, return_state=True)
        self.post_layers = self._to_tuple([layers.TimeDistributed(layers.Dense(lw, activation='relu')) for lw in post_lws])
        self.model_outputs = self._to_tuple([layers.TimeDistributed(layers.Dense(lw, activation=None)) for lw in output_sizes])

        self.input_sizes = self._to_tuple(input_sizes)
        self.output_sizes = self._to_tuple(output_sizes)
        self.lstm_width = self._to_tuple(lstm_lws)
        self._warmup_steps = 1
        self._predict_steps = 1
        self._feedback = {i: -1 for i in range(len(input_sizes))}

        # build model
        # batch size=32, timesteps=5
        self._predict_steps = 5
        self._warmup_steps = 2
        sample_data = [np.repeat(np.arange(10).cat_dist_reshape(-1, 1), 8, axis=1).cat_dist_reshape(10, 8, 1) for in_size in input_sizes]
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        self(sample_data)

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    @property
    def warmup_steps(self):
        return self._warmup_steps

    @warmup_steps.setter
    def warmup_steps(self, new_value):
        self._warmup_steps = new_value

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_value):
        self._predict_steps = new_value

    @property
    def feedback(self):
        return self._feedback

    @feedback.setter
    def feedback(self, new_value):
        self._feedback = new_value

    def _eval_lstm_init(self, data):
        #states_mem = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        states_mem = []
        for l in self.lstm_layers:
            data, *s = l(data)
            #tf.print('before concat: ', tf.shape(s))
            #s = tf.concat(s, axis=0)
            #tf.print('after concat: ', tf.shape(s))
            #states_mem = states_mem.write(states_mem.size(), s)
            states_mem.append(s)
        #return data[:, -1, tf.newaxis], tuple(states_mem)
        #states_mem = states_mem.stack()
        return data[:, -1, np.newaxis], states_mem

    def _eval_lstm(self, data, init_states):
        states_mem = []
        for l, init_l in zip(self.lstm_layers, init_states):
            data, *s = l(data, init_l)
            states_mem.append(s)
        return data, states_mem
        #states_mem = []
        #for l, init_s in zip(self.lstm_layers, init_states):
        #    data, *s = l(data, initial_state=init_s)
        #    states_mem.append(s)
        #return data, tuple(states_mem)

    def evaluate_step(self, inputs, states):
        # wrap to list in case there is only one input stream
        #if type(inputs) is not tuple:
        #    inputs = (inputs, )

        # input branches
        #for inp, in_branch in zip(inputs, self.in_branches):
        #    x = inp
        #    for layer in in_branch:
        #        x = layer(x)
        #    data.append(x)
        # concatenate
        #data = self.concat(data)

        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layers
        if states is None:
            data, states = self._eval_lstm_init(data)
        else:
            data, states = self._eval_lstm(data, states)
        # post lstm layers
        for layer in self.post_layers:
            data = layer(data)
        # output layers
        outputs = tuple([layer(data) for layer in self.model_outputs])
        #outputs = tf.TensorArray(tf.float32, size=len(self.output_sizes))
        #for i_out in range(len(self.model_outputs)):
        #    outputs.write(i_out, self.model_outputs[i_out](data))
        #outputs = outputs.stack()
        #outputs = tf.ragged.stack([l(data) for l in self.model_outputs])

        return outputs, states

    def _detect_feedback(self, feedback, inputs, warmup_steps, predict_steps):
        i_out = 0
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < warmup_steps + predict_steps - 1:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(self.model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {warmup_steps + predict_steps - 1}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_sizes[i_in] != self.output_sizes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match:'))

        return feedback_out

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        if type(inputs) is tuple and len(inputs) == 3:
            inputs = (inputs,)
        elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
            inputs = (inputs,)

        n_warmup = self._warmup_steps
        n_predict = self._predict_steps
        feedback = self._detect_feedback(self._feedback, inputs, n_warmup, n_predict)

        # generate one list per output
        #output_buffer = tf.nest.map_structure(lambda _: tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True), self.model_outputs)
        output_buffer = [[] for _ in self.model_outputs]

        # use only the first self.warmup_steps number of steps for warmup
        # this yields one prediction step
        #warmup_inputs = tf.nest.map_structure(lambda inp: inp[:, 0:n_warmup], inputs)
        warmup_inputs = [inp[:, 0:n_warmup] for inp in inputs]
        #warmup_inputs = tf.ragged.stack([inp[:, 0:n_warmup] for inp in inputs], axis=0)

        last_prediction, states = self.evaluate_step(warmup_inputs, None)
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

        for i_t in range(n_predict - 1):  # warmup yields first prediction, so do one less
            next_input = self._gather_next_input(inputs, last_prediction, i_t + n_warmup, feedback)

            last_prediction, states = self.evaluate_step(next_input, states)
            for i_out, out_buf in enumerate(output_buffer):
                out_buf.append(last_prediction[i_out])

        #output_buffer = tf.nest.map_structure(lambda outp: outp.stack(), output_buffer)
        #output_buffer = tf.nest.map_structure(lambda outp: tf.transpose(outp, [1, 0, 2]), output_buffer)
        #output_buffer = [output_buffer[i] for i in range(output_buffer.shape[0])]
        output_buffer = [tf.concat(outp, axis=1) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    """
    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)
        tf.print([tf.shape(layer) for layer in gradients])
        tf.print(tf.reduce_mean(gradients[0], axis=0))
        tf.print(tf.reduce_mean(gradients[5], axis=0))
        tf.print(tf.reduce_mean(gradients[4], axis=0))
        #tf.print(tf.shape(gradients[4]))
        #tf.print(tf.shape(gradients[5]))
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    """

    @staticmethod
    def _batch_dim(inp):
        if len(np.shape(inp)) >= 2:
            return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) == 3:
            return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        return np.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if np.ndim(inp) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if i_t < self._time_dim(inp):
                inp_curr.append(inp[:, i_t, np.newaxis])  # add back time dimension
                #print(f'timestep {i_t} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])  # has same dimensions as input, so time dimension is present
                #print(f'timestep {i_t} input{i_in}: using feedback')

        return inp_curr


class AutoregressivePredictorWithWarmup(keras.Model):

    def __init__(self, input_sizes, output_sizes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lw=32, **kwargs):
        super(AutoregressivePredictorWithWarmup, self).__init__(**kwargs)

        # input checks
        input_sizes = self._to_list(input_sizes)
        output_sizes = self._to_list(output_sizes)
        if type(pre_lws) is int:
            pre_lws = tuple([(pre_lws,) for inp in input_sizes])
        intermediate_lws = self._to_list(intermediate_lws)
        #lstm_lws = self._to_list(lstm_lws)
        post_lws = self._to_list(post_lws)

        in_branches = []
        for branch_lws in pre_lws:
            in_branches.append([layers.Dense(lw, activation='relu') for lw in branch_lws])
        self.in_branches = self._to_tuple(in_branches)

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate(axis=-1)
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in intermediate_lws])
        self.rec_cell = layers.LSTMCell(lstm_lw)
        self.post_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in post_lws])
        self.model_outputs = self._to_tuple([layers.Dense(lw, activation=None) for lw in output_sizes])

        self.input_sizes = self._to_tuple(input_sizes)
        self.output_sizes = self._to_tuple(output_sizes)
        self.lstm_width = lstm_lw
        self._feedback = {i: -1 for i in range(len(input_sizes))}

        # build model
        #self._predict_steps = 5
        #self._warmup_steps = 2
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        #self(sample_data)

        self._warmup_steps = 1
        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val

    @property
    def warmup_steps(self):
        return self._warmup_steps

    @warmup_steps.setter
    def warmup_steps(self, new_val):
        self._warmup_steps = new_val

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.warmup_steps + self.predict_steps - 1:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(self.model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.warmup_steps + self.predict_steps - 1}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_sizes[i_in] != self.output_sizes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match:'))

        return feedback_out

    def _eval_step(self, inputs, states, training):
        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layer
        data, states = self._run_recurrent_layer(data, states, training)
        # post lstm layers
        for layer in self.post_layers:
            data = layer(data)

        # output layers
        outputs = tuple([layer(data) for layer in self.model_outputs])
        #outputs = tf.TensorArray(tf.float32, size=len(self.output_sizes))
        #for i_out in range(len(self.model_outputs)):
        #    outputs.write(i_out, self.model_outputs[i_out](data))
        #outputs = outputs.stack()
        #outputs = tf.ragged.stack([l(data) for l in self.model_outputs])

        return outputs, states

    def _run_recurrent_layer(self, data, states, training):
        if states is None:
            batch_size = tf.shape(data)[0]
            mock = tf.fill([batch_size, self.lstm_width], 0.0)
            #states = [tf.zeros(shape=(batch_size, self.lstm_width)), tf.zeros(shape=(batch_size, self.lstm_width))]
            states = [mock, mock]

            last_step = data
            for i in range(self._warmup_steps):
                last_step, states = self.rec_cell(data[:, i], states=states, training=training)
            data = last_step
        else:
            data, states = self.rec_cell(data, states=states, training=training)
        return data, states

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        if type(inputs) is tuple and len(inputs) == 3:
            inputs = (inputs,)
        elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
            inputs = (inputs,)

        feedback = self._detect_feedback(self._feedback, inputs)

        #output_buffer = [tf.TensorArray(tf.float32, size=0, dynamic_size=0) for outp in self.output_sizes]
        output_buffer = [[] for _ in self.output_sizes]

        warmup_inputs = [inp[:, 0:self._warmup_steps] for inp in inputs]
        last_prediction, state = self._eval_step(warmup_inputs, None, training)

        self._store_step(last_prediction, output_buffer)

        for i_t in range(self._predict_steps - 1):  # warmup yields first prediction, so do one less
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            #next_inputs = last_prediction
            last_prediction, state = self._eval_step(next_inputs, state, training)
            self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        output_buffer = [tf.transpose(outp, [1, 0, 2]) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[0]
            else:
                return tf.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) == 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return tf.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        return tf.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if len(tf.shape(inp)) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t + self._warmup_steps
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])  # add back time dimension
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])  # has same dimensions as input
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressivePredictor(keras.Model):

    def __init__(self, input_sizes, output_sizes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lw=32, **kwargs):
        super(AutoregressivePredictor, self).__init__(**kwargs)

        # input checks
        input_sizes = self._to_list(input_sizes)
        output_sizes = self._to_list(output_sizes)
        if type(pre_lws) is int:
            pre_lws = tuple([(pre_lws,) for inp in input_sizes])
        intermediate_lws = self._to_list(intermediate_lws)
        #lstm_lws = self._to_list(lstm_lws)
        post_lws = self._to_list(post_lws)

        in_branches = []
        for branch_lws in pre_lws:
            in_branches.append([layers.Dense(lw, activation='relu') for lw in branch_lws])
        self.in_branches = self._to_tuple(in_branches)

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate(axis=-1)
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in intermediate_lws])
        self.rec_cell = layers.LSTMCell(lstm_lw)
        self.post_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in post_lws])
        self.model_outputs = self._to_tuple([layers.Dense(lw, activation=None) for lw in output_sizes])

        self.input_sizes = self._to_tuple(input_sizes)
        self.output_sizes = self._to_tuple(output_sizes)
        self.lstm_width = lstm_lw
        self._feedback = {i: -1 for i in range(len(input_sizes))}

        # build model
        #self._predict_steps = 5
        #self._warmup_steps = 2
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        #self(sample_data)

        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(self.model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_sizes[i_in] != self.output_sizes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match:'))

        return feedback_out

    def _eval_step(self, inputs, states, training):
        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layer
        data, states = self.rec_cell(data, states=states, training=training)
        # post lstm layers
        for layer in self.post_layers:
            data = layer(data)

        # output layers
        outputs = tuple([layer(data) for layer in self.model_outputs])
        #outputs = tf.TensorArray(tf.float32, size=len(self.output_sizes))
        #for i_out in range(len(self.model_outputs)):
        #    outputs.write(i_out, self.model_outputs[i_out](data))
        #outputs = outputs.stack()
        #outputs = tf.ragged.stack([l(data) for l in self.model_outputs])

        return outputs, states

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        if type(inputs) is tuple and len(inputs) == 3:
            inputs = (inputs,)
        elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
            inputs = (inputs,)

        feedback = self._detect_feedback(self._feedback, inputs)

        #output_buffer = [tf.TensorArray(tf.float32, size=0, dynamic_size=0) for outp in self.output_sizes]
        output_buffer = [[] for _ in self.output_sizes]

        # state and last_input dummies that will never actually be used
        batch_size = self._batch_dim(inputs[0])
        #batch_size = tf.shape(inputs[0])[0]
        #mock = tf.fill(tf.TensorShape([batch_size, self.lstm_width]), 0.0)
        mock = tf.zeros(shape=(batch_size, self.lstm_width))
        state = [mock, mock]
        last_prediction = [tf.fill([batch_size, out_size], 0.0) for out_size in self.output_sizes]

        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction, state = self._eval_step(next_inputs, state, training)
            self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        output_buffer = [tf.transpose(outp, [1, 0, 2]) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return tf.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) == 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return tf.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        return tf.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if len(tf.shape(inp)) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressiveProbabilisticPredictor(keras.Model):

    def __init__(self, input_sizes, output_sizes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lw=32, **kwargs):
        super(AutoregressiveProbabilisticPredictor, self).__init__(**kwargs)

        # input checks
        input_sizes = self._to_list(input_sizes)
        output_sizes = self._to_list(output_sizes)
        if type(pre_lws) is int:
            pre_lws = tuple([(pre_lws,) for inp in input_sizes])
        intermediate_lws = self._to_list(intermediate_lws)
        #lstm_lws = self._to_list(lstm_lws)
        post_lws = self._to_list(post_lws)

        in_branches = []
        for branch_lws in pre_lws:
            in_branches.append([layers.Dense(lw, activation='relu') for lw in branch_lws])
        self.in_branches = self._to_tuple(in_branches)

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate(axis=-1)
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in intermediate_lws])
        self.rec_cell = layers.LSTMCell(lstm_lw)
        self.post_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in post_lws])
        #self.model_outputs = self._to_tuple([layers.Dense(lw, activation=None) for lw in output_sizes])
        self.pre_outputs = []
        self.model_outputs = []
        for lw in output_sizes:
            params = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(lw), activation=None)
            outp = tfp.layers.MultivariateNormalTriL(lw)
            self.pre_outputs.append(params)
            self.model_outputs.append(outp)

        self.input_sizes = self._to_tuple(input_sizes)
        self.output_sizes = self._to_tuple(output_sizes)
        self.lstm_width = lstm_lw
        self._feedback = {i: -1 for i in range(len(input_sizes))}

        # build model
        #self._predict_steps = 5
        #self._warmup_steps = 2
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        #self(sample_data)

        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(self.model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_sizes[i_in] != self.output_sizes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match:'))

        return feedback_out

    def _eval_step(self, inputs, states, training):
        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layer
        data, states = self.rec_cell(data, states=states, training=training)
        # post lstm layers
        for layer in self.post_layers:
            data = layer(data)

        # output layers
        #outputs = tuple([layer(data) for layer in self.model_outputs])
        #outputs = tf.TensorArray(tf.float32, size=len(self.output_sizes))
        #for i_out in range(len(self.model_outputs)):
        #    outputs.write(i_out, self.model_outputs[i_out](data))
        #outputs = outputs.stack()
        #outputs = tf.ragged.stack([l(data) for l in self.model_outputs])
        outputs = []
        for pre_outp, outp_distr in zip(self.pre_outputs, self.model_outputs):
            params = pre_outp(data)
            outputs.append(outp_distr(params))

        return outputs, states

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        if type(inputs) is tuple and len(inputs) == 3:
            inputs = (inputs,)
        elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
            inputs = (inputs,)

        feedback = self._detect_feedback(self._feedback, inputs)

        #output_buffer = [tf.TensorArray(tf.float32, size=0, dynamic_size=0) for outp in self.output_sizes]
        output_buffer = [[] for _ in self.output_sizes]

        # state and last_input dummies that will never actually be used
        batch_size = self._batch_dim(inputs[0])
        #batch_size = tf.shape(inputs[0])[0]
        #mock = tf.fill(tf.TensorShape([batch_size, self.lstm_width]), 0.0)
        mock = tf.zeros(shape=(batch_size, self.lstm_width))
        state = [mock, mock]
        last_prediction = [tf.fill([batch_size, out_size], 0.0) for out_size in self.output_sizes]

        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction, state = self._eval_step(next_inputs, state, training)
            self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        output_buffer = [tf.transpose(outp, [1, 0, 2]) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return tf.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) == 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return tf.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        return tf.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if len(tf.shape(inp)) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressiveMultiHeadPredictor(keras.Model):

    def __init__(self, input_sizes, output_sizes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lw=32,
                 num_heads: int = 1, **kwargs):
        super(AutoregressiveMultiHeadPredictor, self).__init__(**kwargs)

        # input checks
        input_sizes = self._to_list(input_sizes)
        output_sizes = self._to_list(output_sizes)
        if type(pre_lws) is int:
            pre_lws = tuple([(pre_lws,) for inp in input_sizes])
        intermediate_lws = self._to_list(intermediate_lws)
        #lstm_lws = self._to_list(lstm_lws)
        post_lws = self._to_list(post_lws)

        in_branches = []
        for branch_lws in pre_lws:
            in_branches.append([layers.Dense(lw, activation='relu') for lw in branch_lws])
        self.in_branches = self._to_tuple(in_branches)

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate(axis=-1)
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in intermediate_lws])
        self.rec_cell = layers.LSTMCell(lstm_lw)
        #self.model_outputs = self._to_tuple([layers.Dense(lw, activation=None) for lw in output_sizes])
        #self.post_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in post_lws])
        self.heads = []
        for _ in range(num_heads):
            #pre_outputs = []
            model_outputs = []
            for lw in output_sizes:
                #params = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(lw), activation=None)
                #outp = tfp.layers.MultivariateNormalTriL(lw)
                #pre_outputs.append(params)
                outp = layers.Dense(lw, activation=None)
                model_outputs.append(outp)
            #self.heads.append((pre_outputs, model_outputs))
            post_layers = self._to_tuple([layers.Dense(lw, activation='relu') for lw in post_lws])
            self.heads.append((post_layers, model_outputs))

        self.input_sizes = self._to_tuple(input_sizes)
        self.output_sizes = self._to_tuple(output_sizes)
        self.lstm_width = lstm_lw
        self._feedback = {i: -1 for i in range(len(input_sizes))}

        # build model
        #self._predict_steps = 5
        #self._warmup_steps = 2
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        #self(sample_data)

        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val
        # re-compile to trigger retracing of training methods to make use of new predict_steps value
        #self.compiled_loss = None
        #self.compiled_metrics = None
        #self.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        post_layers, model_outputs = self.heads[0]
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_sizes[i_in] != self.output_sizes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match:'))

        return feedback_out

    def _eval_step(self, inputs, states, head_indices, training):
        batch_size = self._batch_dim(inputs[0])
        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layer
        data, states = self.rec_cell(data, states=states, training=training)

        head_out_batch_elem_idx = []
        for post_layers, model_outputs in self.heads:
            # post lstm layers
            for layer in post_layers:
                data = layer(data)

            out_batch_elem_idx = []
            for model_output in model_outputs:
                out_batch_elem_idx.append(model_output(data))
            head_out_batch_elem_idx.append(out_batch_elem_idx)
        # bring batch dimension to front to make selection of head per batch item easier
        batch_head_out_elem_idx = tf.transpose(tf.stack(head_out_batch_elem_idx), perm=[2, 0, 1, 3])
        # select per batch item one head
        #print(tf.shape(tf.range(batch_size, dtype=tf.int32)))
        #print(tf.shape(tf.squeeze(head_indices, axis=1)))
        indices_batch_head = tf.stack([tf.range(batch_size, dtype=tf.int32), tf.squeeze(head_indices, axis=1)], axis=1)
        batch_out_elem_idx = tf.gather_nd(batch_head_out_elem_idx, indices_batch_head)
        # bring output index to front again
        out_batch_elem_idx = tf.transpose(batch_out_elem_idx, perm=[1, 0, 2])

        #pre_outputs, model_outputs = self.heads[0]

        return out_batch_elem_idx, states

        # output layers
        #outputs = tuple([layer(data) for layer in self.model_outputs])
        #outputs = tf.TensorArray(tf.float32, size=len(self.output_sizes))
        #for i_out in range(len(self.model_outputs)):
        #    outputs.write(i_out, self.model_outputs[i_out](data))
        #outputs = outputs.stack()
        #outputs = tf.ragged.stack([l(data) for l in self.model_outputs])

        #outputs = []
        #for pre_outp, outp_distr in zip(pre_outputs, model_outputs):
        #    params = pre_outp(data)
        #    outputs.append(outp_distr(params))

        #return outputs, states

    #def __call__(self, *args, **kwargs):
        #kwargs['active_head'] = self.active_head
    #    super().__call__(*args, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        #if type(inputs) is tuple and len(inputs) == 3:
        #    inputs = (inputs,)
        #elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
        #    inputs = (inputs,)

        head_indices = inputs[0]
        inputs = tuple([inp for inp in inputs[1:]])

        feedback = self._detect_feedback(self._feedback, inputs)

        #output_buffer = [tf.TensorArray(tf.float32, size=0, dynamic_size=0) for outp in self.output_sizes]
        output_buffer = [[] for _ in self.output_sizes]

        # state and last_input dummies that will never actually be used
        batch_size = self._batch_dim(inputs[0])
        #batch_size = tf.shape(inputs[0])[0]
        #mock = tf.fill(tf.TensorShape([batch_size, self.lstm_width]), 0.0)
        mock = tf.zeros(shape=(batch_size, self.lstm_width))
        state = [mock, mock]
        last_prediction = [tf.fill([batch_size, out_size], 0.0) for out_size in self.output_sizes]

        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction, state = self._eval_step(next_inputs, state, head_indices, training)
            self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        output_buffer = [tf.transpose(outp, [1, 0, 2]) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

    def OUT_OF_ORDER_train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) == 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        if isinstance(inp, tf.Tensor):
            return tf.shape(inp)[-1]
        else:
            return np.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if len(tf.shape(inp)) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressiveMultiHeadPredictorMk2(keras.Model):

    def __init__(self, input_shapes, output_shapes, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lw=32,
                 num_heads: int = 1, **kwargs):
        super(AutoregressiveMultiHeadPredictorMk2, self).__init__(**kwargs)

        # input checks
        input_shapes = self._wrap_tuple_list(input_shapes)
        output_shapes = self._wrap_tuple_list(output_shapes)
        if type(pre_lws) is int:
            pre_lws = [(pre_lws,) for _ in input_shapes]
        intermediate_lws = self._wrap_list(intermediate_lws)
        post_lws = self._wrap_list(post_lws)

        conv_inputs = [True if len(in_shp) == 3 else False for in_shp in input_shapes]
        conv_outputs = [True if len(out_shp) == 3 else False for out_shp in output_shapes]

        in_branches = []
        for conv_inp, branch_lws in zip(conv_inputs, pre_lws):
            if conv_inp:
                in_branches.append([layers.SeparableConv2D(lw, activation='relu', kernel_size=3) for lw in branch_lws])
                in_branches[-1].append(layers.Flatten())
            else:
                in_branches.append([layers.Dense(lw, activation='relu') for lw in branch_lws])
        self.in_branches = in_branches

        # if there are multiple inputs, concatenate; else remove list
        if len(self.in_branches) > 1:
            self.concat = layers.Concatenate(axis=-1)
        else:
            self.concat = layers.Lambda(lambda x: x[0])

        self.intermediate_layers = [layers.Dense(lw, activation='relu') for lw in intermediate_lws]
        self.rec_cell = layers.LSTMCell(lstm_lw)
        self.heads = []
        for _ in range(num_heads):
            post_layers = [layers.Dense(lw, activation='relu') for lw in post_lws]
            pre_outputs = []
            model_outputs = []
            for conv_outp, lw in zip(conv_outputs, output_shapes):
                if conv_outp:
                    pre_outp = layers.Dense(lw[0] * lw[1] * lw[2], activation=None)
                    outp = layers.Conv2DTranspose(lw[2], kernel_size=(1, 1), activation=None)
                else:
                    pre_outp = layers.Lambda(lambda x: x)
                    outp = layers.Dense(lw[0], activation=None)
                pre_outputs.append(pre_outp)
                model_outputs.append(outp)
            self.heads.append((post_layers, pre_outputs, model_outputs))

        self.lstm_width = lstm_lw
        self._feedback = {i: -1 for i in range(len(input_shapes))}
        self._conv_inputs = conv_inputs
        self._conv_outputs = conv_outputs
        self._input_shapes = input_shapes
        self._output_shapes = output_shapes

        # build model
        #self._predict_steps = 5
        #self._warmup_steps = 2
        #sample_data = [np.zeros((42, 5, in_size)) for in_size in input_sizes]
        #self(sample_data)

        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val
        # re-compile to trigger retracing of training methods to make use of new predict_steps value
        #self.compiled_loss = None
        #self.compiled_metrics = None
        #self.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    def _wrap_list(self, x, level=0):
        """
        If x is an int, float or tuple, wraps x into a list. If x is a list itself, recursively wraps all contents
        of x into lists unless an int, float or tuple is encountered.
        :param x: structure to wrap
        :param level: internal use only
        :return: wrapped structure
        """
        if isinstance(x, Iterable) and not type(x) is tuple and not type(x) is str:
            return [self._wrap_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float or type(x) is tuple) and level == 0:
            return [x]
        else:
            return x

    def _wrap_tuple_list(self, x, level=0):
        if type(x) is list:
            return [self._wrap_tuple_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float or type(x) is str) and level == 0:
            return [(x,)]
        elif type(x) is tuple and level == 0:
            return [x]
        elif type(x) is tuple:
            return x
        else:
            return (x,)

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        post_layers, pre_outputs, model_outputs = self.heads[0]
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self._input_shapes[i_in] != self._output_shapes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match: {self._input_shapes[i_in]} vs {self._output_shapes[i_out]}'))

        return feedback_out

    def _eval_step(self, inputs, states, head_indices, training):
        batch_size = self._batch_dim(inputs[0])
        data = []

        for i_in, inp in enumerate(inputs):
            x = inp
            for l in self.in_branches[i_in]:
                x = l(x)
            data.append(x)
        data = self.concat(data)
        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)
        # lstm layer
        data, states = self.rec_cell(data, states=states, training=training)

        head_out_batch_elem_idx = []
        for post_layers, pre_outputs, model_outputs in self.heads:
            data_head = data

            # post lstm layers
            for layer in post_layers:
                data_head = layer(data_head)

            out_batch_elem_idx = []
            for conv_outp, pre_outp, model_output, out_shp in zip(self._conv_outputs, pre_outputs, model_outputs, self._output_shapes):
                if conv_outp:
                    outp = pre_outp(data_head)
                    outp = tf.reshape(outp, (batch_size, *out_shp))
                    outp = model_output(outp)
                    outp = tf.reshape(outp, (batch_size, out_shp[0] * out_shp[1] * out_shp[2]))  # flatten image outputs for now
                else:
                    outp = model_output(data_head)
                out_batch_elem_idx.append(outp)
            head_out_batch_elem_idx.append(out_batch_elem_idx)
        # bring batch dimension to front to make selection of head per batch item easier
        batch_head_out_elem_idx = tf.transpose(tf.stack(head_out_batch_elem_idx), perm=[2, 0, 1, 3])
        # select per batch item one head
        #print(tf.shape(tf.range(batch_size, dtype=tf.int32)))
        #print(tf.shape(tf.squeeze(head_indices, axis=1)))
        indices_batch_head = tf.stack([tf.range(batch_size, dtype=tf.int32), tf.squeeze(head_indices, axis=1)], axis=1)
        batch_out_elem_idx = tf.gather_nd(batch_head_out_elem_idx, indices_batch_head)
        # bring output index to front again
        out_batch_elem_idx = tf.transpose(batch_out_elem_idx, perm=[1, 0, 2])
        # un-flatten image outputs
        out_batch_elem_idx_reshaped = [tf.reshape(out_batch_elem_idx[i], (batch_size, *self._output_shapes[i])) for i in range(len(self._output_shapes))]

        #pre_outputs, model_outputs = self.heads[0]

        return out_batch_elem_idx_reshaped, states


    #def __call__(self, *args, **kwargs):
    #kwargs['active_head'] = self.active_head
    #    super().__call__(*args, **kwargs)


    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        #if type(inputs) is tuple and len(inputs) == 3:
        #    inputs = (inputs,)
        #elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
        #    inputs = (inputs,)

        head_indices = inputs[0]
        inputs = tuple([inp for inp in inputs[1:]])

        feedback = self._detect_feedback(self._feedback, inputs)

        #output_buffer = [tf.TensorArray(tf.float32, size=0, dynamic_size=0) for outp in self.output_sizes]
        output_buffer = [[] for _ in self._output_shapes]

        # state and last_input dummies that will never actually be used
        batch_size = self._batch_dim(inputs[0])
        #batch_size = tf.shape(inputs[0])[0]
        #mock = tf.fill(tf.TensorShape([batch_size, self.lstm_width]), 0.0)
        mock = tf.zeros(shape=(batch_size, self.lstm_width))
        state = [mock, mock]
        last_prediction = [tf.fill([batch_size, *out_shp], 0.0) for out_shp in self._output_shapes]

        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction, state = self._eval_step(next_inputs, state, head_indices, training)
            self._store_step(last_prediction, output_buffer)

        # now ordering is (time, batch, data), reorder to (batch, time, data)
        output_buffer = [tf.stack(outp) for outp in output_buffer]
        output_buffer = [tf.transpose(outp, [1, 0, 2, 3, 4]) if conv_out
                         else tf.transpose(outp, [1, 0, 2])
                         for outp, conv_out in zip(output_buffer, self._conv_outputs)]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

    def OUT_OF_ORDER_train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        # tf.rank(inp) does not work if there are None(s) in the shape of inp
        if len(tf.shape(inp)) > 2:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        if isinstance(inp, tf.Tensor):
            return tf.shape(inp)[-1]
        else:
            return np.shape(inp)[-1]

    @staticmethod
    def _has_time_dim(inp):
        if len(tf.shape(inp)) == 3:
            return True
        else:
            return False

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressiveMultiHeadFullyConvolutionalPredictor_(keras.Model):

    def __init__(self, state_shape, action_shape, pre_lws=32, intermediate_lws=64, post_lws=32, lstm_lw=32,
                 num_heads: int = 1, **kwargs):
        super(AutoregressiveMultiHeadFullyConvolutionalPredictor, self).__init__(**kwargs)

        assert len(state_shape) == 3 or len(state_shape) == 1, f'State shape should be 1D or 3D, but is {len(state_shape)}D'

        self.input_shapes = self._to_tuple([state_shape, action_shape])
        self.output_shapes = (tuple(state_shape), )

        if type(pre_lws) is int:
            pre_lws = [[pre_lws], [pre_lws]]
        else:
            assert len(pre_lws) == 2, 'Provide either one int or layer widths for state and action preprocessing branch'
            pre_lws = self._to_list(pre_lws)
        intermediate_lws = self._to_list(intermediate_lws)
        post_lws = self._to_list(post_lws)

        self.conv_layers = len(state_shape) == 3

        state_constr = partial(layers.Conv2D, kernel_size=3, padding='SAME') if len(state_shape) == 3 else layers.Dense
        layer_constr = partial(layers.Conv2D, kernel_size=5, padding='SAME') if len(state_shape) == 3 else layers.Dense
        lstm_constr = partial(layers.ConvLSTM2D, kernel_size=5, padding='SAME', return_state=True) if len(state_shape) == 3 else layers.LSTMCell

        self.state_branch = [state_constr(lw, activation='relu') for lw in pre_lws[0]]
        self.action_branch = [layers.Dense(lw, activation='relu') for lw in pre_lws[1]]
        self.act_reshape = InflateLayer(state_shape[0:2]) if len(state_shape) == 3 else layers.Lambda(lambda x: x)

        self.concat = layers.Concatenate(axis=-1)

        self.intermediate_layers = [layer_constr(lw, activation='relu') for lw in intermediate_lws]
        self.rec_cell = lstm_constr(lstm_lw)

        self.heads = []
        for _ in range(num_heads):
            post_layers = [layer_constr(lw) for lw in post_lws]
            out_layers = [state_constr(state_shape[-1], activation=None)]  # x, y of conv layers never changed during whole network, so we only need to make number of channels fit
            self.heads.append((post_layers, out_layers))

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.lstm_width = lstm_lw
        self._feedback = {i: -1 for i in range(2)}
        self._predict_steps = 1


    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val
        # re-compile to trigger retracing of training methods to make use of new predict_steps value
        #self.compiled_loss = None
        #self.compiled_metrics = None
        #self.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

    def _to_tuple(self, x, level=0):
        if isinstance(x, Iterable):
            return tuple([self._to_tuple(elem, level + 1) for elem in x])
        elif (type(x) is int or type(x) is float) and level == 0:
            return (x,)
        else:
            return x

    def _to_list(self, x, level=0):
        if isinstance(x, Iterable):
            return [self._to_list(elem, level + 1) for elem in x]
        elif (type(x) is int or type(x) is float) and level == 0:
            return [x]
        else:
            return x

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        post_layers, model_outputs = self.heads[0]
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > len(model_outputs):
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_shapes[i_in] != self.output_shapes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match: {self.input_shapes[i_in]} vs {self.output_shapes[i_out]}'))

        return feedback_out

    def _eval_step(self, inputs, states, head_indices, training):
        batch_size = self._batch_dim(inputs[0])

        # preprocessing branches
        x_s, x_a = inputs
        for l in self.state_branch:
            x_s = l(x_s)
        for l in self.action_branch:
            x_a = l(x_a)

        data = self.concat([x_s, self.act_reshape(x_a)])

        # pre lstm layers
        for layer in self.intermediate_layers:
            data = layer(data)

        # lstm layer
        if self.conv_layers:
            data = tf.expand_dims(data, 1)
            data, *states = self.rec_cell(data, initial_state=states, training=training)
        else:
            data, states = self.rec_cell(data, states=states, training=training)

        # NOTE: variable naming is dimension indices connected with underscores
        head_out_batch_elem_idx = []
        for post_layers, model_outputs in self.heads:
            # post lstm layers
            for layer in post_layers:
                data = layer(data)

            out_batch_elem_idx = []
            for model_output in model_outputs:
                out_batch_elem_idx.append(model_output(data))
            head_out_batch_elem_idx.append(out_batch_elem_idx)

        # bring batch dimension to front to make selection of head per sample easier
        if self.conv_layers:
            batch_head_out_elem_idx = tf.transpose(tf.stack(head_out_batch_elem_idx), perm=[2, 0, 1, 3, 4, 5])
        else:
            batch_head_out_elem_idx = tf.transpose(tf.stack(head_out_batch_elem_idx), perm=[2, 0, 1, 3])

        #print(tf.shape(tf.range(batch_size, dtype=tf.int32)))
        #print(tf.shape(tf.squeeze(head_indices, axis=1)))
        # select one head per sample
        i_h = tf.stack([tf.range(batch_size, dtype=tf.int32), tf.squeeze(head_indices, axis=1)], axis=1)
        batch_out_elem_idx = tf.gather_nd(batch_head_out_elem_idx, i_h)

        # bring output index to front again
        if self.conv_layers:
            out_batch_elem_idx = tf.transpose(batch_out_elem_idx, perm=[1, 0, 2, 3, 4])
        else:
            out_batch_elem_idx = tf.transpose(batch_out_elem_idx, perm=[1, 0, 2])

        return out_batch_elem_idx, states

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        #if type(inputs) is tuple and len(inputs) == 3:
        #    inputs = (inputs,)
        #elif isinstance(inputs, tf.Tensor) and len(inputs.get_shape()) is 3:
        #    inputs = (inputs,)

        head_indices = inputs[0]
        inputs = tuple([inp for inp in inputs[1:]])

        feedback = self._detect_feedback(self._feedback, inputs)

        #output_buffer = [tf.TensorArray(tf.float32, size=0, dynamic_size=0) for outp in self.output_sizes]
        output_buffer = [[] for _ in self.output_shapes]

        # state and last_input dummies that will never actually be used
        batch_size = self._batch_dim(inputs[0])
        if self.conv_layers:
            mock = tf.zeros(shape=(batch_size, self.input_shapes[0][0], self.input_shapes[0][1], self.lstm_width))
        else:
            mock = tf.zeros(shape=(batch_size, self.lstm_width))

        state = [mock, mock]
        last_prediction = [tf.fill([batch_size, *out_size], 0.0) for out_size in self.output_shapes]

        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction, state = self._eval_step(next_inputs, state, head_indices, training)
            self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        output_buffer = [tf.transpose(outp, [1, 0, 2, 3, 4]) if self.conv_layers else
                         tf.transpose(outp, [1, 0, 2])
                         for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

    def OUT_OF_ORDER_train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        if len(tf.shape(inp)) >= 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        if len(tf.shape(inp)) > 3:
            raise ValueError(f'Data seems to have more than one dimension: {tf.shape(inp)}')

        if isinstance(inp, tf.Tensor):
            return tf.shape(inp)[-1]
        else:
            return np.shape(inp)[-1]

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressiveMultiHeadFullyConvolutionalPredictor(keras.Model):

    def __init__(self, state_shape, n_actions, common_filters=64, intermediate_filters=64, per_head_filters=32,
                 n_heads=1, **kwargs):
        assert len(state_shape) == 3, f'State shape must be 3D, but is {len(state_shape)}D'

        super(AutoregressiveMultiHeadFullyConvolutionalPredictor, self).__init__(**kwargs)

        self.input_shapes = (tuple(state_shape), (1,))
        self.output_shapes = (tuple(state_shape), )
        self.rec_layers = []

        # transformation submodel for action inputs
        in_action = layers.Input((1,))
        x_action = InflateActionLayer(state_shape[:2], n_actions)(in_action)
        self.transform_action = keras.Model(inputs=in_action, outputs=x_action)

        # for concatenation of state and action inputs
        self.concat = layers.Concatenate(axis=-1)

        # recurrent layer(s)
        self.common_rec_layers = [StatefulConvLSTM2DCell(common_filters, kernel_size=3)]
        self.rec_layers.extend(self.common_rec_layers)

        # intermediate layer(s)
        #self.intermediate_layers = [ResidualConv2D(intermediate_filters)]

        self.heads = []
        for _ in range(n_heads):
            post_rec_layers = [StatefulConvLSTM2DCell(per_head_filters, kernel_size=3)]
            self.rec_layers.extend(post_rec_layers)
            # x, y of conv layers never changed during whole network, so we only need to make number of channels fit
            out_state = layers.Conv2D(state_shape[-1], kernel_size=3, padding='SAME', activation=None)
            out_reward = None
            self.heads.append((post_rec_layers, out_state, out_reward))

        self.state_shape = state_shape
        self._feedback = {i: -1 for i in range(2)}
        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val
        # re-compile to trigger retracing of training methods to make use of new predict_steps value
        #self.compiled_loss = None
        #self.compiled_metrics = None
        #self.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        post_layers, out_state, out_reward = self.heads[0]
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > 1:
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_shapes[i_in] != self.output_shapes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match: {self.input_shapes[i_in]} vs {self.output_shapes[i_out]}'))

        return feedback_out

    def _eval_step(self, inputs, head_indices, training):
        batch_size = self._batch_dim(inputs[0])

        # preprocessing branches
        x_s, x_a = inputs
        x_a = self.transform_action(x_a)[:, 0, ...]
        x = self.concat([x_s, x_a])

        # recursive layers
        for layer in self.common_rec_layers:
            x = layer(x)

        # conv layers with skip connections and batch norm
        #for layer in self.intermediate_layers:
        #    x = layer(x, training=training)

        # NOTE: variable naming is dimension indices connected with underscores
        head_out_batch_elem_idx = []
        for post_layers, out_state, out_reward in self.heads:
            # post lstm layers
            x_h = x
            for layer in post_layers:
                x_h = layer(x_h)

            # append reward as well later
            head_out_batch_elem_idx.append([out_state(x_h)])

        # bring batch dimension to front to make selection of head per sample easier
        batch_head_out_elem_idx = tf.transpose(tf.stack(head_out_batch_elem_idx), perm=[2, 0, 1, 3, 4, 5])

        # select one head per sample
        i_h = tf.stack([tf.range(batch_size, dtype=tf.int32), tf.squeeze(head_indices, axis=1)], axis=1)
        batch_out_elem_idx = tf.gather_nd(batch_head_out_elem_idx, i_h)

        # bring output index to front again
        out_batch_elem_idx = tf.transpose(batch_out_elem_idx, perm=[1, 0, 2, 3, 4])

        return out_batch_elem_idx

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        batch_size = self._batch_dim(inputs[0])

        for l in self.rec_layers:
            l.reset_state()

        head_indices = inputs[0]
        inputs = tuple([inp for inp in inputs[1:]])

        feedback = self._detect_feedback(self._feedback, inputs)
        output_buffer = [[] for _ in self.output_shapes]

        last_prediction = [tf.fill([batch_size, *out_size], 0.0) for out_size in self.output_shapes]
        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction = self._eval_step(next_inputs, head_indices, training)
            output_buffer = self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        # currently, outputs are ordered like (timestep, batch, width, height, channels)
        # they need to be transposed to (batch, timestep, width, height, channels)
        output_buffer = [tf.transpose(outp, [1, 0, 2, 3, 4]) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])
        return output_buffer

    def OUT_OF_ORDER_train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        if len(tf.shape(inp)) >= 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        if len(tf.shape(inp)) > 3:
            raise ValueError(f'Data seems to have more than one dimension: {tf.shape(inp)}')

        if isinstance(inp, tf.Tensor):
            return tf.shape(inp)[-1]
        else:
            return np.shape(inp)[-1]

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class AutoregressiveFullyConvolutionalPredictor(keras.Model):

    def __init__(self, state_shape, n_actions, common_filters=64, intermediate_filters=64, per_head_filters=32,
                 n_heads=1, **kwargs):
        assert len(state_shape) == 3, f'State shape must be 3D, but is {len(state_shape)}D'

        super(AutoregressiveFullyConvolutionalPredictor, self).__init__(**kwargs)

        self.input_shapes = (tuple(state_shape), (1,))
        self.output_shapes = (tuple(state_shape), )
        self.rec_layers = []

        # transformation submodel for action inputs
        in_action = layers.Input((1,))
        x_action = InflateActionLayer(state_shape[:2], n_actions)(in_action)
        self.transform_action = keras.Model(inputs=in_action, outputs=x_action)

        # for concatenation of state and action inputs
        self.concat = layers.Concatenate(axis=-1)

        # recurrent layer(s)
        self.common_rec_layers = [StatefulConvLSTM2DCell(common_filters, kernel_size=3)]
        self.rec_layers.extend(self.common_rec_layers)

        # intermediate layer(s)
        #self.intermediate_layers = [ResidualConv2D(intermediate_filters)]

        self.heads = []
        for _ in range(n_heads):
            post_rec_layers = [StatefulConvLSTM2DCell(per_head_filters, kernel_size=3)]
            self.rec_layers.extend(post_rec_layers)
            # x, y of conv layers never changed during whole network, so we only need to make number of channels fit
            out_state = layers.Conv2D(state_shape[-1], kernel_size=3, padding='SAME', activation=None)
            out_reward = None
            self.heads.append((post_rec_layers, out_state, out_reward))

        self.state_shape = state_shape
        self._feedback = {i: -1 for i in range(2)}
        self._predict_steps = 1

    @property
    def predict_steps(self):
        return self._predict_steps

    @predict_steps.setter
    def predict_steps(self, new_val):
        self._predict_steps = new_val
        # re-compile to trigger retracing of training methods to make use of new predict_steps value
        #self.compiled_loss = None
        #self.compiled_metrics = None
        #self.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        post_layers, out_state, out_reward = self.heads[0]
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > 1:
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_shapes[i_in] != self.output_shapes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match: {self.input_shapes[i_in]} vs {self.output_shapes[i_out]}'))

        return feedback_out

    def _eval_step(self, inputs, head_indices, training):
        batch_size = self._batch_dim(inputs[0])

        # preprocessing branches
        x_s, x_a = inputs
        x_a = self.transform_action(x_a)[:, 0, ...]
        x = self.concat([x_s, x_a])

        # recursive layers
        for layer in self.common_rec_layers:
            x = layer(x)

        # conv layers with skip connections and batch norm
        #for layer in self.intermediate_layers:
        #    x = layer(x, training=training)

        post_layers, out_state, out_reward = self.heads[0]
        # post lstm layers
        for layer in post_layers:
            x = layer(x)

        return [out_state(x)]

    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        batch_size = self._batch_dim(inputs[0])

        for l in self.rec_layers:
            l.reset_state()

        head_indices = inputs[0]
        inputs = tuple([inp for inp in inputs[1:]])

        feedback = self._detect_feedback(self._feedback, inputs)
        output_buffer = [[] for _ in self.output_shapes]

        last_prediction = [tf.fill([batch_size, *out_size], 0.0) for out_size in self.output_shapes]
        for i_t in range(self._predict_steps):
            next_inputs = self._gather_next_input(inputs, last_prediction, i_t, feedback)
            last_prediction = self._eval_step(next_inputs, head_indices, training)
            output_buffer = self._store_step(last_prediction, output_buffer)

        output_buffer = [tf.stack(outp) for outp in output_buffer]
        # currently, outputs are ordered like (timestep, batch, width, height, channels)
        # they need to be transposed to (batch, timestep, width, height, channels)
        output_buffer = [tf.transpose(outp, [1, 0, 2, 3, 4]) for outp in output_buffer]

        if len(output_buffer) == 1:
            output_buffer = output_buffer[0]

        return output_buffer

    def _store_step(self, last_prediction, output_buffer):
        for i_out, out_buf in enumerate(output_buffer):
            out_buf.append(last_prediction[i_out])

        return output_buffer

    def OUT_OF_ORDER_train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_weights)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        if len(tf.shape(inp)) >= 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        if len(tf.shape(inp)) > 3:
            raise ValueError(f'Data seems to have more than one dimension: {tf.shape(inp)}')

        if isinstance(inp, tf.Tensor):
            return tf.shape(inp)[-1]
        else:
            return np.shape(inp)[-1]

    def _gather_next_input(self, inputs, outp_last, i_t, feedback):
        offset = i_t
        inp_curr = []

        for i_in, inp in enumerate(inputs):
            if offset < self._time_dim(inp):
                inp_curr.append(inp[:, offset])
                #print(f'timestep {offset} input{i_in}: using available input')
            else:
                i_out = feedback[i_in]
                inp_curr.append(outp_last[i_out])
                #print(f'timestep {offset} input{i_in}: using feedback')

        return inp_curr


class _AutoregressiveProbabilisticFullyConvolutionalPredictor(keras.Model):

    def __init__(self, observation_shape, n_actions, vqvaq_codes_sampler, n_cb_vectors, common_filters=128, intermediate_filters=64,
                 per_head_filters=128, n_heads=1, **kwargs):
        assert len(observation_shape) == 2, f'Expecting (w, h) shaped cb vector index matrices, got {len(observation_shape)}D'

        super(AutoregressiveProbabilisticFullyConvolutionalPredictor, self).__init__(**kwargs)

        def transform_indices(inp):
            indices = tf.argmax(inp, axis=-1)
            inp = tf.stop_gradient(vqvaq_codes_sampler(indices))
            return inp

        vqvae_i_to_vec = layers.Lambda(lambda inp: transform_indices(inp))
        self.rec_layers = []

        # deterministic model to form state belief h_t = f(o_t-1, a_t-1, c_t-1)
        # note: h_t-1 is injected into the model not as explicit input but through previous LSTM state c_t-1
        in_o = layers.Input((*observation_shape, n_cb_vectors))
        in_a = layers.Input((1,))
        o_cb_vectors = vqvae_i_to_vec(in_o)
        a_inflated = InflateActionLayer(observation_shape, n_actions)(in_a)
        h = layers.Concatenate(axis=-1)([o_cb_vectors, a_inflated])  # note: use cb vector indices and cb vectors
        h_rec_1 = StatefulConvLSTM2DCell(common_filters, kernel_size=3)(h)
        h_rec_2 = StatefulConvLSTM2DCell(common_filters, kernel_size=3)(h_rec_1)
        #h_rec_1 = layers.ConvLSTM2D(common_filters, kernel_size=3, return_sequences=True)(h)
        #h_rec_2 = layers.ConvLSTM2D(common_filters, kernel_size=3, return_sequences=True)(h_rec_1)
        self.det_model = keras.Model(inputs=[in_o, in_a], outputs=h_rec_2)

        # keep track of all recurrent layers to do reset after episode
        #self.rec_layers.extend(self.det_model.layers[-2:])

        # stochastic model to implement p(o_t+1 | o_t, a_t, h_t)
        in_params_o = layers.Input((*observation_shape, common_filters))
        x_params_o = layers.Conv2D(intermediate_filters, kernel_size=3, padding='SAME', activation='relu')(in_params_o)
        x_params_o = layers.Conv2D(intermediate_filters, kernel_size=3, padding='SAME', activation='relu')(x_params_o)
        x_params_o = layers.Conv2D(n_cb_vectors, kernel_size=3, padding='SAME', activation=None, name='p_o')(x_params_o)
        self.params_o_model = keras.Model(inputs=in_params_o, outputs=x_params_o)

        # stochastic model to implement p(r_t+1 | o_t, a_t, h_t)
        in_params_r = layers.Input((*observation_shape, common_filters))
        x_params_r = layers.Flatten()(in_params_r)
        x_params_r = layers.Dense(intermediate_filters, activation='relu')(x_params_r)
        x_params_r = layers.Dense(2, activation=None, name='p_r')(x_params_r)
        self.params_r_model = keras.Model(inputs=in_params_r, outputs=x_params_r)

        """
        self.heads = []
        for _ in range(n_heads):
            n_out_channels = n_cb_vectors

            rec_layer = StatefulConvLSTM2DCell(per_head_filters, kernel_size=3)
            self.rec_layers.append(rec_layer)

            # TODO: use softmax activation for the last convolutional layer?
            post_layers = [rec_layer, layers.Conv2D(n_out_channels, kernel_size=3, padding='SAME', activation='relu')]

            out_state = tfp.layers.DistributionLambda(
                #make_distribution_fn=lambda t: tfd.Independent(tfd.RelaxedOneHotCategorical(1, logits=tf.math.softmax(t)),
                #                                               reinterpreted_batch_ndims=2),
                make_distribution_fn=lambda t: tfd.RelaxedOneHotCategorical(1, logits=t),
                convert_to_tensor_fn=lambda t: t.sample(),
            )

            out_reward = None
            self.heads.append((post_layers, out_state, out_reward))
        """

        self.observation_shape = observation_shape
        self.action_shape = (1,)
        self.reward_shape = (1,)

        #import datetime
        #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.summary_writer = tf.summary.create_file_writer(f'./logs/{current_time}')
        self._train_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._n_cb_vectors = n_cb_vectors

        self._obs_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='crossentropy_error')
        self._rew_accuracy = tf.keras.metrics.MeanSquaredError(name='mse')
        self._loss_tracker = tf.keras.metrics.Mean(name='loss')

    """
    def _detect_feedback(self, feedback, inputs):
        i_out = 0
        feedback_out = copy.deepcopy(feedback)
        for i_in in range(len(inputs)):
            if self._time_dim(inputs[i_in]) < self.predict_steps:
                if feedback_out[i_in] == -1:
                    feedback_out[i_in] = i_out
                    i_out += 1
            else:
                feedback_out[i_in] = -1

        if i_out > 1:
            raise ValueError(('Too many inputs need feedback during prediction, but there are not enough outputs '
                              'available. To avoid wrong automatic feedback detection, please provide a list '
                              'to the feedback property of the predictor. '
                              f'Timesteps per input: {[self._time_dim(inputs[i]) for i in range(len(inputs))]}, '
                              f'required timesteps to not activate feedback is {self.predict_steps}'))

        for i_in, i_out in feedback_out.items():
            if i_out == -1:
                continue
            if self.input_shapes[i_in] != self.output_shapes[i_out]:
                raise ValueError((f'Input no {i_in} was assigned to receive feedback from output no. {i_out} but '
                                  f'their shapes do not match: {self.input_shapes[i_in]} vs {self.output_shapes[i_out]}'))

        return feedback_out
    """

    #@tf.function
    def _eval_step(self, o_inp, a_inp, i_head, training):
        h = self.det_model([o_inp, a_inp], training=training)
        params_o = self.params_o_model(h)
        params_r = self.params_r_model(h)

        o_pred = tfd.RelaxedOneHotCategorical(1, params_o)
        r_pred = tfd.Normal(loc=params_r[:, 0], scale=params_r[:, 1])
        r_sample = r_pred.sample()
        return o_pred.sample(), r_sample

    #@tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None, **kwargs):
        for l in self.rec_layers:
            l.reset_state()

        i_head, o_in, a_in = inputs
        n_batch = self._batch_dim(i_head)
        n_warmup = tf.shape(o_in)[1]
        n_predict = tf.shape(a_in)[1]

        feedback = n_warmup if n_warmup < n_predict else -1
        o_out = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        r_out = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        o_pred, r_pred = tf.fill([n_batch, *self.observation_shape, self._n_cb_vectors], 0.0), tf.fill([n_batch, 1], 0.0)
        for i_t in range(n_predict):
            #tf.print(f'Processing step {i_t}')
            o_next, a_next = self._next_input(o_in, a_in, o_pred, i_t, feedback)
            o_pred, r_pred = self._eval_step(o_next, a_next, i_head, training)
            o_out = o_out.write(i_t, o_pred)
            r_out = r_out.write(i_t, r_pred)

        #o_out = tf.stack(o_out)
        #r_out = tf.stack(r_out)
        # currently, outputs are ordered like (timestep, batch, width, height)
        # they need to be transposed to (batch, timestep, width, height)
        o_out_transp = tf.transpose(o_out.stack(), [1, 0, 2, 3, 4])
        r_out_transp = tf.transpose(r_out.stack(), [1, 0])

        #with self.summary_writer.as_default():
        #    tf.summary.histogram('0 input indices', tf.reshape(inputs[0][0], [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('1 first prediction indices', tf.reshape(o_out[0][0], [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('2 second prediction indices', tf.reshape(o_out[1][0], [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('3 third prediction indices', tf.reshape(o_out[2][0], [-1]), self._train_step, self._n_cb_vectors)
        #    self._train_step.assign(self._train_step + 1)

        #tf.print('Done call procedure')

        return o_out_transp, tf.expand_dims(r_out_transp, axis=-1)

    #@tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_weights)
            #y_pred = self(x, training=True)
            #loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            o_pred, r_pred = self(x, training=True)
            #o_pred_reshaped = tf.reshape(o_pred, (-1, self._n_cb_vectors))
            #r_pred_reshaped = tf.reshape(r_pred, (-1, 1))
            #o_target = tf.reshape(y[0], (-1, self._n_cb_vectors))
            #r_target = tf.reshape(y[1], (-1, 1))
            #obseravtion_error = tf.losses.categorical_crossentropy(o_target, o_pred_reshaped)
            #reward_error = tf.losses.mean_squared_error(r_target, r_pred_reshaped)
            observation_error = tf.losses.categorical_crossentropy(y[0], o_pred)
            reward_error = tf.losses.mean_squared_error(y[1], r_pred)
            loss = tf.reduce_mean(observation_error) + tf.reduce_mean(reward_error)
            #loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            #loss = - y_hat.log_prob(y)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y, (o_pred, r_pred), sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        #return {m.name: m.result() for m in self.metrics}

        self._obs_accuracy.update_state(y[0], o_pred)
        self._rew_accuracy.update_state(y[1], r_pred)
        self._loss_tracker.update_state(loss)

        return {'loss': self._loss_tracker.result(),
                'observation_error': self._obs_accuracy.result(),
                'reward_error': self._rew_accuracy.result()}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    @staticmethod
    def _time_dim(inp):
        if len(tf.shape(inp)) >= 3:
            if isinstance(inp, tf.Tensor):
                return inp.get_shape()[1]
            else:
                return np.shape(inp)[1]
        else:
            raise ValueError(f'{len(np.shape(inp))}D structure found, can\'t infer time dimension')

    @staticmethod
    def _data_dim(inp):
        if len(tf.shape(inp)) > 3:
            raise ValueError(f'Data seems to have more than one dimension: {tf.shape(inp)}')

        if isinstance(inp, tf.Tensor):
            return tf.shape(inp)[-1]
        else:
            return np.shape(inp)[-1]

    def _next_input(self, o_in, a_in, o_last, i_t, feedback):
        if i_t < feedback:
            return o_in[:, i_t], a_in[:, i_t]
        else:
            return o_last, a_in[:, i_t]


class AutoregressiveProbabilisticFullyConvolutionalPredictor(keras.Model):

    def __init__(self, observation_shape, n_actions, vqvae_codes_sampler, n_cb_vectors, vae_latent_shape,
                 common_filters=384, intermediate_filters=96, per_head_filters=128, n_models=1, **kwargs):
        assert len(observation_shape) == 2, f'Expecting (w, h) shaped cb vector index matrices, got {len(observation_shape)}D'

        if kwargs.pop('debug_log', False):
            import datetime
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(f'./logs/{current_time}')
        else:
            self.summary_writer = None

        super(AutoregressiveProbabilisticFullyConvolutionalPredictor, self).__init__(**kwargs)

        self.observation_shape = observation_shape
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self._train_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._n_cb_vectors = n_cb_vectors
        self._vae_latent_shape = vae_latent_shape
        self._h_out_shape = (*observation_shape, common_filters)

        self.det_model = self._det_state(common_filters, n_actions, n_cb_vectors, observation_shape, vqvae_codes_sampler, vae_latent_shape)
        self.params_o_model = self._params_o(self._h_out_shape, intermediate_filters, n_cb_vectors)
        self.params_r_model = self._params_r(self._h_out_shape, intermediate_filters)


        #self.det_model = self._det_state(common_filters, n_actions, n_cb_vectors, observation_shape, vqvae_codes_sampler, vae_latent_shape)
        #self.params_o_model = self._params_o(self._h_out_shape, intermediate_filters, n_cb_vectors)
        #self.params_r_model = self._params_r(self._h_out_shape, intermediate_filters)
        #self._open_loop_rollout = tf.Variable(False, dtype=tf.bool, trainable=False)

        """
        self.heads = []
        for _ in range(n_heads):
            n_out_channels = n_cb_vectors

            rec_layer = StatefulConvLSTM2DCell(per_head_filters, kernel_size=3)
            self.rec_layers.append(rec_layer)

            # TODO: use softmax activation for the last convolutional layer?
            post_layers = [rec_layer, layers.Conv2D(n_out_channels, kernel_size=3, padding='SAME', activation='relu')]

            out_state = tfp.layers.DistributionLambda(
                #make_distribution_fn=lambda t: tfd.Independent(tfd.RelaxedOneHotCategorical(1, logits=tf.math.softmax(t)),
                #                                               reinterpreted_batch_ndims=2),
                make_distribution_fn=lambda t: tfd.RelaxedOneHotCategorical(1, logits=t),
                convert_to_tensor_fn=lambda t: t.sample(),
            )

            out_reward = None
            self.heads.append((post_layers, out_state, out_reward))
        """

        self._obs_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='crossentropy_error')
        self._rew_accuracy = tf.keras.metrics.MeanSquaredError(name='mse')
        self._loss_tracker = tf.keras.metrics.Mean(name='loss')

    def _params_pred(self, h_out_shape, n_mdl):
        in_params_pred = layers.Input(h_out_shape, name='p_pred_in')
        x_params_pred = layers.Flatten()(in_params_pred)
        x_params_pred = layers.Dense(64, activation='relu')(x_params_pred)
        x_params_pred = layers.Dense(32, activation='relu', name='p_pred_out')(x_params_pred)

        return keras.Model(inputs=in_params_pred, outputs=x_params_pred, name='p_pred_model')

    def _params_r(self, h_out_shape, intermediate_filters):
        # stochastic model to implement p(r_t+1 | o_t, a_t, h_t)
        in_params_r = layers.Input(h_out_shape, name='p_r_in')
        x_params_r = layers.Flatten()(in_params_r)
        x_params_r = layers.Dense(intermediate_filters, activation='relu')(x_params_r)
        x_params_r = layers.BatchNormalization()(x_params_r)
        x_params_r = layers.Dense(2, activation=None, name='p_r_out')(x_params_r)

        return keras.Model(inputs=in_params_r, outputs=x_params_r, name='p_r_model')

    def _params_o(self, h_out_shape, intermediate_filters, n_cb_vectors):
        # stochastic model to implement p(o_t+1 | o_t, a_t, h_t)
        in_params_o = layers.Input(h_out_shape, name='p_o_in')
        x_params_o = layers.Conv2D(intermediate_filters, kernel_size=5, padding='SAME', activation='relu')(in_params_o)
        x_params_o = layers.BatchNormalization()(x_params_o)
        x_params_o = layers.Conv2D(intermediate_filters, kernel_size=3, padding='SAME', activation='relu')(x_params_o)
        x_params_o = layers.BatchNormalization()(x_params_o)
        x_params_o = layers.Conv2D(n_cb_vectors, kernel_size=3, padding='SAME', activation=None, name='p_o_out')(
            x_params_o)

        return keras.Model(inputs=in_params_o, outputs=x_params_o, name='p_o_model')

    def _det_state(self, common_filters, n_actions, n_cb_vectors, s_obs, vqvae_codes_sampler, s_vae_latent):
        # deterministic model to form state belief h_t = f(o_t-1, a_t-1, c_t-1)
        # note: h_t-1 is injected into the model not as explicit input but through previous LSTM state c_t-1
        vqvae_i_to_vec = self._index_transform_layer(s_obs, vqvae_codes_sampler, s_vae_latent)
        in_o = layers.Input((None, *s_obs, n_cb_vectors), name='h_o_in')
        in_a = layers.Input((None, 1), name='h_a_in')
        lstm_c = layers.Input((*s_obs, common_filters), name='h_lstm_in0')
        lstm_h = layers.Input((*s_obs, common_filters), name='h_lstm_in1')
        o_cb_vectors = vqvae_i_to_vec(in_o)
        a_inflated = InflateActionLayer(s_obs, n_actions, True)(in_a)
        h = layers.Concatenate(axis=-1)([o_cb_vectors, a_inflated])
        h = layers.Conv2D(common_filters, kernel_size=5, padding='SAME', activation='relu')(h)
        h = layers.BatchNormalization()(h)
        h = layers.Conv2D(common_filters, kernel_size=3, padding='SAME', activation='relu')(h)
        h = layers.BatchNormalization()(h)
        h_rec_1, *h_states_1 = layers.ConvLSTM2D(common_filters, kernel_size=3, return_state=True,
                                                 return_sequences=True, padding='SAME', name='h_out')(h, initial_state=[
            lstm_c, lstm_h])

        return keras.Model(inputs=[in_o, in_a, lstm_c, lstm_h], outputs=[h_rec_1, h_states_1], name='h_model')

    def _index_transform_layer(self, observation_shape, vqvae_codes_sampler, vae_latent_shape):
        def transform_indices(inp):
            inp_shape = tf.shape(inp)

            indices = tf.argmax(inp, axis=-1)
            # fold timesteps into batch dimension
            reshaped = tf.reshape(indices, (-1, *observation_shape))
            codes = tf.stop_gradient(vqvae_codes_sampler(reshaped))
            # fold timesteps back into time dimension
            codes_reshaped = tf.reshape(codes, (inp_shape[0], inp_shape[1], *vae_latent_shape))

            return codes_reshaped

        return layers.Lambda(lambda inp: transform_indices(inp))

    @property
    def open_loop_rollout(self):
        return self._open_loop_rollout

    @open_loop_rollout.setter
    def open_loop_rollout(self, new_val):
        self.open_loop_rollout.assign(new_val)

    def _temp(self):
        #return (tf.nn.tanh(self._temperature) + 1) * 2
        return 0.01

    @tf.function
    def _open_loop_step(self, o_inp, a_inp, i_head, lstm_states, training):
        n_batch = tf.shape(a_inp)[0]

        h, lstm_states = self.det_model([o_inp, a_inp] + lstm_states, training=training)
        h_flattened = tf.reshape(h, (-1, *self._h_out_shape))
        params_o = self.params_o_model(h_flattened, training=training)
        params_o = tf.reshape(params_o, (n_batch, 1, *self.observation_shape, self._n_cb_vectors))
        params_r = self.params_r_model(h_flattened, training=training)
        params_r = tf.reshape(params_r, (n_batch, 1, 2))

        o_pred = tfd.RelaxedOneHotCategorical(self._temp(), params_o).sample()
        r_pred = tfd.Normal(loc=params_r[..., 0, tf.newaxis], scale=params_r[..., 1, tf.newaxis]).sample()
        return o_pred, r_pred, lstm_states

    @tf.function
    def _rollout_closed_loop(self, inputs, mask=None, training=None, **kwargs):
        i_head, o_in, a_in = inputs
        n_batch = tf.shape(a_in)[0]
        n_time = tf.shape(a_in)[1]

        n_warmup = tf.shape(o_in)[1]
        n_predict = tf.shape(a_in)[1]

        tf.debugging.assert_equal(n_warmup, n_predict, ('For closed loop rollout, observations and actions have to be '
                                                        f'provided in equal numbers, but are {n_warmup} and {n_predict}'))

        dummy_states = [tf.fill((n_batch, *self._h_out_shape), 0.0), tf.fill((n_batch, *self._h_out_shape), 0.0)]
        h, states = self.det_model([o_in, a_in] + dummy_states, training=training)
        h_flattened = tf.reshape(h, (-1, *self._h_out_shape))

        params_o = self.params_o_model(h_flattened, training=training)
        params_o = tf.reshape(params_o, (n_batch, n_time, *self.observation_shape, self._n_cb_vectors))
        params_r = self.params_r_model(h_flattened, training=training)
        params_r = tf.reshape(params_r, (n_batch, n_time, 2))

        o_pred = tfd.RelaxedOneHotCategorical(self._temp(), params_o).sample()
        r_pred = tfd.Normal(loc=params_r[..., 0, tf.newaxis], scale=params_r[..., 1, tf.newaxis]).sample()

        #with self.summary_writer.as_default():
        #    tf.summary.histogram('0 input indices', tf.reshape(tf.argmax(o_in[0][0], axis=-1), [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('1 first prediction indices', tf.reshape(tf.argmax(o_pred[0][0], axis=-1), [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('2 second prediction indices', tf.reshape(tf.argmax(o_pred[1][0], axis=-1), [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('3 third prediction indices', tf.reshape(tf.argmax(o_pred[2][0], axis=-1), [-1]), self._train_step, self._n_cb_vectors)
        #    self._train_step.assign(self._train_step + 1)

        return o_pred, r_pred

    @tf.function
    def call(self, inputs, mask=None, training=None):
        if self._open_loop_rollout:
            trajectories = self._rollout_open_loop(inputs, mask, training)
        else:
            trajectories = self._rollout_closed_loop(inputs, mask, training)
        return trajectories


    @tf.function
    def _rollout_open_loop(self, inputs, mask=None, training=None, **kwargs):
        i_head, o_in, a_in = inputs

        n_batch = self._batch_dim(i_head)
        n_warmup = tf.shape(o_in)[1]
        n_predict = tf.shape(a_in)[1]

        tf.debugging.assert_less(n_warmup, n_predict, ('For rollout, less observations than actions are expected, '
                                                       f'but I got {n_warmup} observation and {n_predict} action '
                                                       f'steps.'))

        t_start_feedback = n_warmup
        o_out = tf.TensorArray(tf.float32, size=0, dynamic_size=True, name='o_store')
        r_out = tf.TensorArray(tf.float32, size=0, dynamic_size=True, name='r_store')

        lstm_states = [tf.fill((n_batch, *self._h_out_shape), 0.0, name='lstm_init_0'), tf.fill((n_batch, *self._h_out_shape), 0.0, name='lstm_init_1')]
        o_pred, r_pred = tf.fill([n_batch, 1, *self.observation_shape, self._n_cb_vectors], 0.0), tf.fill([n_batch, 1, 1], 0.0)
        for i_t in range(n_predict):
            #tf.print(f'Processing step {i_t}')
            o_next, a_next = self._next_input(o_in, a_in, o_pred, i_t, t_start_feedback)
            o_pred, r_pred, lstm_states = self._open_loop_step(o_next, a_next, i_head, lstm_states, training)
            o_out = o_out.write(i_t, o_pred[:, 0])
            r_out = r_out.write(i_t, r_pred[:, 0])

        #o_out = tf.stack(o_out)
        #r_out = tf.stack(r_out)
        # currently, outputs are ordered like (timestep, batch, width, height, one_hot_vec)
        # they need to be transposed to (batch, timestep, width, height, one_hot_vec)
        o_out_transp = tf.transpose(o_out.stack(), [1, 0, 2, 3, 4])
        r_out_transp = tf.transpose(r_out.stack(), [1, 0, 2])

        #with self.summary_writer.as_default():
        #    tf.summary.histogram('0 input indices', tf.reshape(inputs[0][0], [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('1 first prediction indices', tf.reshape(o_out[0][0], [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('2 second prediction indices', tf.reshape(o_out[1][0], [-1]), self._train_step, self._n_cb_vectors)
        #    tf.summary.histogram('3 third prediction indices', tf.reshape(o_out[2][0], [-1]), self._train_step, self._n_cb_vectors)
        #    self._train_step.assign(self._train_step + 1)

        #tf.print('Done call procedure')

        return o_out_transp, r_out_transp #tf.expand_dims(r_out_transp, axis=-1)

    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        #tf.print(self.active_head)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_weights)
            #y_pred = self(x, training=True)
            #loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            if self.summary_writer:
                with self.summary_writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=True)
                    o_pred, r_pred = self(x, training=True)
                    tf.summary.trace_export(name='Convolutional_Predictor_Trace', step=self._train_step.value(), profiler_outdir='graph')
            else:
                o_pred, r_pred = self(x, training=True)
            #o_pred_reshaped = tf.reshape(o_pred, (-1, self._n_cb_vectors))
            #r_pred_reshaped = tf.reshape(r_pred, (-1, 1))
            #o_target = tf.reshape(y[0], (-1, self._n_cb_vectors))
            #r_target = tf.reshape(y[1], (-1, 1))
            #obseravtion_error = tf.losses.categorical_crossentropy(o_target, o_pred_reshaped)
            #reward_error = tf.losses.mean_squared_error(r_target, r_pred_reshaped)
            observation_error = tf.losses.categorical_crossentropy(y[0], o_pred)
            reward_error = tf.losses.mean_squared_error(y[1], r_pred)
            loss = tf.reduce_mean(observation_error) + tf.reduce_mean(reward_error)
            #loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            #loss = - y_hat.log_prob(y)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        """
        recorded_grads = []
        recorded_grads.append(tape.gradient(loss, self.layers[3].trainable_weights))
        tf.summary.experimental.set_step(self._train_steps)
        self._train_steps += 1
        with self.writer.as_default():
            for i, g in enumerate(recorded_grads):
                curr_grad = g[0]

                tf.summary.scalar(f'grad_mean_layer{i+1}', tf.reduce_mean(tf.abs(curr_grad)))
                tf.summary.histogram(f'grad_histogram_layer{i+1}', curr_grad)
        self.writer.flush()
        """
        # clip gradients
        #gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y, (o_pred, r_pred), sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        #return {m.name: m.result() for m in self.metrics}

        self._obs_accuracy.update_state(y[0], o_pred)
        self._rew_accuracy.update_state(y[1], r_pred)
        self._loss_tracker.update_state(loss)

        self._train_step.assign(self._train_step.value() + 1)

        return {'loss': self._loss_tracker.result(),
                'observation_error': self._obs_accuracy.result(),
                'reward_error': self._rew_accuracy.result(),
                't': self._temp()}

    @staticmethod
    def _batch_dim(inp):
        if len(tf.shape(inp)) >= 2:
            if isinstance(inp, tf.Tensor):
                return tf.shape(inp)[0]
            else:
                return np.shape(inp)[0]
        else:
            raise ValueError('1D array found, can\'t infer batch dimension')

    def _next_input(self, o_in, a_in, o_last, i_t, feedback):
        if i_t < feedback:
            return o_in[:, i_t, tf.newaxis], a_in[:, i_t, tf.newaxis]
        else:
            return o_last, a_in[:, i_t, tf.newaxis]


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
        x_params_pred = layers.Dense(64, activation='relu')(x_params_pred)
        x_params_pred, *lstm_states = layers.LSTM(decider_lw, return_state=True, return_sequences=True)(x_params_pred, initial_state=[lstm_c, lstm_h])
        x_params_pred = layers.Dense(n_mdl, activation=None, name='p_pred_out')(x_params_pred)

        return keras.Model(inputs=[in_o, lstm_c, lstm_h], outputs=[x_params_pred, lstm_states], name='p_pred_model')

    def _gen_params_r(self, h_out_shape, prob_filters):
        # stochastic model to implement p(r_t+1 | o_t, a_t, h_t)
        in_h = layers.Input(h_out_shape, name='p_r_in')
        x_params_r = layers.Flatten()(in_h)
        x_params_r = layers.Dense(prob_filters, activation='relu')(x_params_r)
        #x_params_r = layers.LayerNormalization()(x_params_r)
        x_params_r = layers.Dense(prob_filters, activation='relu')(x_params_r)
        #x_params_r = layers.LayerNormalization()(x_params_r)
        x_params_r = layers.Dense(2, activation=None, name='p_r_out')(x_params_r)

        return keras.Model(inputs=in_h, outputs=x_params_r, name='p_r_model')

    def _gen_params_o(self, h_out_shape, prob_filters, vqvae):
        # stochastic model to implement p(o_t+1 | o_t, a_t, h_t)
        in_h = layers.Input(h_out_shape, name='p_o_in')
        x_params_o = layers.Conv2D(prob_filters, kernel_size=5, padding='SAME', activation='relu')(in_h)
        #x_params_o = layers.LayerNormalization()(x_params_o)
        x_params_o = layers.Conv2D(prob_filters, kernel_size=3, padding='SAME', activation='relu')(x_params_o)
        #x_params_o = layers.LayerNormalization()(x_params_o)
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
        #h = layers.LayerNormalization()(h)
        h = layers.Conv2D(det_filters, kernel_size=3, padding='SAME', activation='relu')(h)
        #h = layers.LayerNormalization()(h)
        h, *h_states = layers.ConvLSTM2D(det_filters, kernel_size=3, return_state=True,
                                                 return_sequences=True, padding='SAME',
                                                 name='h_out')(h, initial_state=[lstm_c, lstm_h])
        #h = layers.Conv2D(det_filters, kernel_size=3, padding='SAME', activation=None)(h)

        return keras.Model(inputs=[in_o, in_a, lstm_c, lstm_h], outputs=[h, h_states, a_inflated], name='h_model')

    def _gen_index_transform_layer(self, vqvae):
        def transform_fun(input):
            indices = tf.argmax(input, -1)
            index_matrices = tf.stop_gradient(vqvae.indices_to_embeddings(indices))
            return index_matrices

        return tf.keras.layers.Lambda(lambda inp: transform_fun(inp))

    def _temp_predictor_picker(self, training):
        if training:
            return 2
        else:
            return 0.01

    def _temp(self, training):
        if training:
            return 2
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
    def _rollout_open_loop(self, inputs, training=None):
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
            trajectories = self._rollout_open_loop(inputs, training)
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
