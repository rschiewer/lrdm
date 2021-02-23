import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from abc import abstractmethod
from tf_tools import ImageResizeLayer, Sampling


class ConvolutionalVaeBase(keras.Model):

    def __init__(self, input_shape, latent_size, *args, **kwargs):
        assert len(input_shape) == 3, 'Expected 3D input shape width * height * channels'
        assert input_shape[2] == 1 or input_shape[2] == 3, 'Expected number of channels to be 1 or 3'
        # assert type(in_spec) is layers.InputSpec
        # assert in_spec.is_compatible_with(tf.TensorSpec(None, None, None), tf.float32)

        super(ConvolutionalVaeBase, self).__init__(*args, **kwargs)

        self.encoder = self._generate_encoder(input_shape, latent_size)
        self.decoder = self._generate_decoder(input_shape, latent_size)

        enc_input_shape = tf.nest.map_structure(lambda tensor: tensor.shape, self.encoder.inputs)
        enc_outputs = self.encoder.compute_output_shape(enc_input_shape)
        assert len(enc_outputs) == 3, 'Encoder should have three outputs: z_mean, z_log_var, z'
        assert all([outp.is_compatible_with(tf.TensorShape([None, latent_size])) for outp in enc_outputs])
        self._set_inputs(self.encoder.inputs)  # to set input_shape of this model, doesn't seem to work automatically

        # use tensor objects here in case the model is serialized
        self.in_shape = input_shape
        self.latent_size = latent_size

    @abstractmethod
    def _generate_encoder(self, in_shape, base_depth, latent_size) -> keras.Model:
        pass

    @abstractmethod
    def _generate_decoder(self, out_shape, base_depth, latent_size) -> keras.Model:
        pass

    def get_config(self):
        config = {'in_shape': self.in_shape, 'latent_size': self.latent_size, 'base_depth': self.base_depth}
        return config
        # try:
        #    base_config = super(ConvolutionalVaeBase, self).get_config()
        # except NotImplementedError:
        #    base_config = {}
        # return dict(list(base_config.items()) + list(config.items()))

    def encode(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if np.ndim(data) - len(self.in_shape) == 2:
            time_dimension = True
        else:
            time_dimension = False

        batch_size = data.shape[0]
        timesteps = None
        if time_dimension:
            timesteps = data.shape[1]
            data = data.reshape(-1, *self.in_shape)

        encoded, _, _ = self.encoder.predict(data)

        if time_dimension:
            encoded = encoded.cat_dist_reshape(batch_size, timesteps, self.latent_size)

        return encoded

    def decode(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if np.ndim(data) == 3:
            time_dimension = True
        else:
            time_dimension = False

        batch_size = data.shape[0]
        timesteps = None
        if time_dimension:
            timesteps = data.shape[1]
            data = data.reshape(-1, self.latent_size)

        encoded = self.decoder.predict(data)

        if time_dimension:
            encoded = encoded.cat_dist_reshape(batch_size, timesteps, *self.in_shape)

        return encoded


class BetaVAE(ConvolutionalVaeBase):

    def __init__(self, input_shape, latent_size=16, beta=2, kl_loss_factor=0.001, kl_loss_factor_mul=1.003, **kwargs):
        super(BetaVAE, self).__init__(input_shape, latent_size, **kwargs)

        # use tensor objects here in case the model is serialized
        self.kl_loss_factor = tf.Variable(kl_loss_factor, trainable=False)
        self.kl_loss_factor_mul = tf.Variable(kl_loss_factor_mul, trainable=False)
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32)
        self._normally_trainable_layers = set()

    def _generate_encoder(self, in_shape, latent_size):
        enc_input = keras.Input(in_shape, name='encoder_input')
        x = ImageResizeLayer((64, 64))(enc_input)
        x = layers.Conv2D(16, 3, 2, activation='relu')(x)
        x = layers.Conv2D(32, 3, 2, activation='relu')(x)
        x = layers.Conv2D(16, 3, 2, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        z_mean = keras.layers.Dense(latent_size, name='z_mean')(x)
        z_log_var = keras.layers.Dense(latent_size, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs=enc_input, outputs=[z_mean, z_log_var, z], name='encoder')
        return encoder

    def _generate_decoder(self, out_shape, latent_size):
        dec_input = keras.Input(shape=(latent_size,), name='decoder_input')
        x = layers.Dense(7 * 7 * 32, activation='relu')(dec_input)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.Conv2DTranspose(64, 3, 2, activation='relu')(x)
        x = layers.Conv2DTranspose(32, 3, 2, activation='relu')(x)
        x = layers.Conv2DTranspose(3, 4, 2)(x)  # kernel size of 4 to produce 64x64 sized images as output
        dec_output = ImageResizeLayer(out_shape[0:2])(x)
        decoder = keras.Model(inputs=dec_input, outputs=dec_output, name='decoder')
        return decoder

    def train_step(self, data):
        sample_weight = None
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
            elif len(data) == 3:
                x, y, sample_weight = data
        else:
            x, y = data, data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(y - reconstruction))
            reconstruction_loss *= np.prod(self.in_shape)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss *= self.kl_loss_factor * self.beta
            total_loss = reconstruction_loss + kl_loss  # self.beta * self.latent_size / np.prod(self.in_shape) * kl_loss
            for loss_tensor in self.losses:
                total_loss += loss_tensor
        grads = tape.gradient(total_loss, self.trainable_weights)
        # grads = [(tf.clip_by_value(grad, -0.5, 0.5)) for grad in grads]
        # tf.debugging.check_numerics(grads, 'grads contain inf or nan')
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # if self.kl_loss_factor < 1:
        #    self.kl_loss_factor.assign(self.kl_loss_factor.value() * 1.003)
        # else:
        #    self.kl_loss_factor.assign(1)
        tf.cond(self.kl_loss_factor < 1,
                lambda: self.kl_loss_factor.assign(self.kl_loss_factor * 1.003),
                lambda: self.kl_loss_factor.assign(1))

        self.add_metric(total_loss, 'loss')
        self.add_metric(reconstruction_loss, 'rec_loss')
        self.add_metric(kl_loss, 'kl_loss')
        self.add_metric(self.kl_loss_factor, 'kl_factor')

        #return {
        #    "loss": total_loss,
        #    "rec_loss": reconstruction_loss,
        #    "kl_loss": kl_loss,
        #    'kl_factor': self.kl_loss_factor
        #}
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, reconstruction, sample_weight=sample_weight)
        self.compiled_metrics.update_state
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        sample_weight = None
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
            elif len(data) == 3:
                x, y, sample_weight = data
        else:
            x, y = data, data

        y_pred = self(x, training=False)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def freeze_params(self):
        for layer in self.encoder.layers:
            if layer.trainable:
                layer.trainable = False
            else:
                self._normally_trainable_layers.add(layer)

        for layer in self.decoder.layers:
            if layer.trainable:
                layer.trainable = False
            else:
                self._normally_trainable_layers.add(layer)

        # self.compile(self.optimizer, metrics=self.metrics)

    def unfreeze_params(self):
        for layer in self.encoder.layers:
            if layer in self._normally_trainable_layers:
                layer.trainable = True
        for layer in self.decoder.layers:
            if layer in self._normally_trainable_layers:
                layer.trainable = True

        # self.compile(self.optimizer, metrics=self.metrics)

    # def get_config(self):
    #    config = {'in_shape': self.in_shape, 'latent_size': self.latent_size, 'base_depth': self.base_depth}
    #    base_config = super(BetaVAE, self).get_config()
    #    return base_config


class LargeBetaVAE(BetaVAE):

    def _generate_encoder(self, in_shape, latent_size):
        enc_input = keras.Input(in_shape, name='encoder_input')
        x = ImageResizeLayer((64, 64))(enc_input)
        x = layers.Conv2D(64, 3, 2, activation='relu')(x)
        x = layers.Conv2D(96, 3, 2, activation='relu')(x)
        x = layers.Conv2D(64, 3, 1, activation='relu')(x)
        x = layers.Conv2D(32, 3, 1, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        z_mean = keras.layers.Dense(latent_size, name='z_mean')(x)
        z_log_var = keras.layers.Dense(latent_size, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs=enc_input, outputs=[z_mean, z_log_var, z], name='encoder')
        return encoder

    def _generate_decoder(self, out_shape, latent_size):
        dec_input = keras.Input(shape=(latent_size,), name='decoder_input')
        x = layers.Dense(11 * 11 * 32, activation='relu')(dec_input)
        x = layers.Reshape((11, 11, 32))(x)
        x = layers.Conv2DTranspose(64, 3, 1, activation='relu')(x)
        x = layers.Conv2DTranspose(96, 3, 1, activation='relu')(x)
        x = layers.Conv2DTranspose(64, 3, 2, activation='relu')(x)
        x = layers.Conv2DTranspose(3, 4, 2)(x)  # kernel size of 4 to produce 64x64 sized images as output
        dec_output = ImageResizeLayer(out_shape[0:2])(x)
        decoder = keras.Model(inputs=dec_input, outputs=dec_output, name='decoder')
        return decoder