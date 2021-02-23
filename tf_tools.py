import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ImageResizeLayer(keras.layers.Layer):

    def __init__(self, out_shape, **kwargs):
        assert np.ndim(out_shape) is 1 and len(out_shape) is 2, 'Expected 2D tuple for out_shape (img_dim_0, img_dim_1)'
        super().__init__(**kwargs)

        self.out_shape = tf.tuple(out_shape)
        self.ratios = None
        self.trainable = False

    def build(self, input_shape):
        assert np.ndim(input_shape) is 1 and len(input_shape) is 4, \
            'Expected 4D tensor input (batch_size, image_dim_0, image_dim_1, channels), ' \
            'but input is: {}'.format(input_shape)
        assert 1 <= input_shape[3] <= 3, 'Expected 1 to 3 channels'

        self.ratios = tf.cast(self.out_shape, tf.float32) / tf.cast(input_shape[1:3], tf.float32)

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, self.out_shape)

    def compute_output_shape(self, input_shape):
        return self.out_shape

    def get_config(self):
        config = {'ratios': self.ratios, 'out_shape': self.out_shape}
        base_config = super(ImageResizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(layers.Layer):

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class InflateLayer(layers.Layer):

    def __init__(self, inflate_dims, **kwargs):
        super(InflateLayer, self).__init__(**kwargs)
        self.trainable = False
        self.inflate_dims = tuple(inflate_dims)

    def call(self, inputs, **kwargs):
        if len(tf.shape(inputs)) == 2:
            #reshaped = tf.reshape(inputs, (batch_dim, 1, 1, data_dim))
            # tf.reshape statement results in None dimension for channels, nobody except tf devs knows why
            reshaped = tf.expand_dims(inputs, 1)
            reshaped = tf.expand_dims(reshaped, 1)
            inflated = tf.tile(reshaped, [1, self.inflate_dims[0], self.inflate_dims[1], 1])
        elif len(tf.shape(inputs)) == 3:

            reshaped = tf.expand_dims(inputs, 2)
            reshaped = tf.expand_dims(reshaped, 2)
            inflated = tf.tile(reshaped, [1, 1, self.inflate_dims[0], self.inflate_dims[1], 1])
        else:
            raise RuntimeError('Inputs are expected to be 2D (batch, data) or 3D (batch, time, data)')

        return inflated


class TimeAwareFlatten(layers.Layer):

    def __init__(self, **kwargs):
        super(TimeAwareFlatten, self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs, **kwargs):
        assert len(tf.shape(inputs)) == 3, 'Inputs are expected to be 3D (batch, time, data)'
        batch_dim = tf.shape(inputs)[0]
        time_dim = tf.shape(inputs)[1]
        reshaped = tf.reshape(inputs, (batch_dim, time_dim, -1))

        return reshaped


class ResidualConv2D(tf.keras.Model):

    def __init__(self, filters, strides=1, data_format='channels_last', **kwargs):
        super(ResidualConv2D, self).__init__(**kwargs)

        assert data_format in ['channels_first', 'channels_last']

        self.n_channels = filters
        self.strides = strides
        self.data_format = data_format

        self.conv_0 = layers.Conv2D(filters, padding='same', kernel_size=3, strides=strides, data_format=data_format,
                                    name='conv_0')
        self.conv_1 = layers.Conv2D(filters, padding='same', kernel_size=3, data_format=data_format, name='conv_1')
        if strides != 1 and strides != (1, 1):  # we need a reshape for the skip connection
            self.conv_2 = layers.Conv2D(filters, padding='same', kernel_size=3, strides=strides,
                                        data_format=data_format, name='conv_2')
        else:
            self.conv_2 = None

        self.bn_0 = layers.BatchNormalization()
        self.bn_1 = layers.BatchNormalization()

    def build(self, input_shape):
        n_input_channels = input_shape[-1] if self.data_format is 'channels_last' else input_shape[-3]

        # if input and output number of channels mismatch, skip connection needs to have channels adjusted
        if n_input_channels != self.n_channels and self.conv_2 is None:
            self.conv_2 = layers.Conv2D(self.n_channels, padding='same', kernel_size=3, strides=self.strides,
                                        data_format=self.data_format)

    def call(self, x, **kwargs):
        training = kwargs.pop('training')
        y = tf.keras.activations.relu(self.bn_0(self.conv_0(x), training=training))
        y = self.bn_1(self.conv_1(y), training=training)
        if self.conv_2 is not None:
            x = self.conv_2(x)
        out = tf.keras.activations.relu(y + x)
        return out

    def get_config(self):
        pass


class InflateActionLayer(layers.Layer):

    def __init__(self, inflate_dims, n_actions, time_dim=False, **kwargs):
        """
        Expects one or more scalar inputs and generates an inflate_dims[0] * inflate_dims[1] * n_actions output tensor
        per input. Each scalar in the input is transformed to a one-hot vector. The one-hot vector is interpreded as a
        1 * 1 * n_actions tensor and resized by copying inflate_dims[0] times along the 0th axis and inflate_dims[1]
        times along the 1st axis. The result is an image with inflate_dims dimensions and one channel per action.
        :param inflate_dims: First and second dimension of the resulting expanded inputs.
        :param n_actions: Third dimension of the resulting expanded inputs.
        :param kwargs: See tf.keras.layers.Layer.
        """
        assert len(inflate_dims) == 2, 'Inflate dimensions must be list or tuple of length 2'

        super(InflateActionLayer, self).__init__(**kwargs)

        self.trainable = False
        self.inflate_dims = inflate_dims
        self.n_actions = n_actions
        self.time_dim = time_dim

    def call(self, inputs, **kwargs):
        # TODO: if ... else for inputs with and without time dimension

        n_batch = tf.shape(inputs)[0]
        if self.time_dim:
            n_time = tf.shape(inputs)[1]
        else:
            n_time = 1

        # inputs might have time dimension
        flattened_inputs = tf.reshape(inputs, (-1, 1))
        # actions batch is shape (batch, 1), one_hot encoding is (batch, 1, one_hot_vec), so remove middle dimension
        one_hot_inputs = tf.one_hot(tf.cast(flattened_inputs, tf.int32), self.n_actions)[..., 0, :]

        def inflate_fn(one_hot_vector):
            # prepend width and height
            #print(one_hot_vector.get_shape())
            reshaped = tf.expand_dims(one_hot_vector, 0)
            #print(reshaped.get_shape())
            reshaped = tf.expand_dims(reshaped, 0)
            #print(reshaped.get_shape())
            # copy along width and height
            inflated = tf.tile(reshaped, [self.inflate_dims[0], self.inflate_dims[1], 1])
            #print(inflated.get_shape())
            return inflated

        inflated = tf.map_fn(inflate_fn, one_hot_inputs)
        if self.time_dim:
            new_shape = tf.concat([tf.expand_dims(n_batch, axis=0), tf.expand_dims(n_time, axis=0), self.inflate_dims, tf.expand_dims(self.n_actions, axis=0)], axis=0)
        else:
            new_shape = tf.concat([tf.expand_dims(n_batch, axis=0), self.inflate_dims, tf.expand_dims(self.n_actions, axis=0)], axis=0)
        inflated_reshaped = tf.reshape(inflated, new_shape)
        return inflated_reshaped


class StatefulConvLSTM2DCell(layers.Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        """
        Returns only the last timestep output. Handles statefulness internally.
        """
        padding = kwargs.pop('padding', 'SAME')

        super(StatefulConvLSTM2DCell, self).__init__()

        self.filters = filters
        self.nested_layer = tf.keras.layers.ConvLSTM2D(filters, kernel_size=kernel_size, padding=padding,
                                                       return_state=True, **kwargs)
        self.states = None

    def reset_state(self):
        #tf.print('resetting states')
        self.states = None

    def call(self, inputs, **kwargs):
        if len(tf.shape(inputs)) == 4:  # add time dimension if none is present
            inputs = tf.expand_dims(inputs, 1)
        # providing 'None' argument for initial_state works as well, so no dummy first state has to be created
        #if self.states is None:
        #    in_shape = (tf.shape(inputs)[0], tf.shape(inputs)[2], tf.shape(inputs)[3])
        #    self.states = [tf.zeros(shape=(*in_shape,  self.filters)), tf.zeros(shape=(*in_shape,  self.filters))]

        output, *self.states = self.nested_layer(inputs, initial_state=self.states, **kwargs)

        return output


class OneHotToIndex(layers.Layer):

    def __init__(self, **kwargs):
        super(OneHotToIndex, self).__init__(**kwargs)
        self.trainable = False

        self.selector_vector = None

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.selector_vector = tf.range(n_features, dtype = tf.float32)

    def call(self, inputs, **kwargs):
        n_features = tf.shape(inputs)[-1]
        n_vectors_total = tf.math.reduce_prod(tf.shape(inputs)[:-1])
        shape_final = tf.shape(inputs)[:-1]

        inputs_reshaped = tf.reshape(inputs, (-1, n_features))
        selector_matrix = []
        for _ in range(n_vectors_total):
            selector_matrix.append(self.selector_vector)
        selector_matrix = tf.stack(selector_matrix)
        #selector_matrix = tf.stack([self.selector_vector for _ in range(n_vectors_total)])
        selected = tf.multiply(inputs_reshaped, selector_matrix)
        selected_indices = tf.reduce_sum(selected, axis=1)
        selected_indices_reshaped = tf.reshape(selected_indices, shape_final)

        return selected_indices_reshaped

