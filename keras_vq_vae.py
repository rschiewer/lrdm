import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tf_tools import maybe_unflatten_time, maybe_flatten_time


class ExponentialMovingAverage(tfkl.Layer):
    """Maintains an exponential moving average for a value.

    Note this module uses debiasing by default. If you don't want this please use
    an alternative implementation.

    This module keeps track of a hidden exponential moving average that is
    initialized as a vector of zeros which is then normalized to give the average.
    This gives us a moving average which isn't biased towards either zero or the
    initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

    Initially:

        hidden_0 = 0

    Then iteratively:

        hidden_i = (hidden_{i-1} - value) * (1 - decay)
        average_i = hidden_i / (1 - decay^i)

    Attributes:
      average: Variable holding average. Note that this is None until the first
        value is passed.
    """

    def __init__(self, decay, name=None):
        """Creates a debiased moving average module.

        Args:
          decay: The decay to use. Note values close to 1 result in a slow decay
            whereas values close to 0 result in faster decay, tracking the input
            values more closely.
          name: Name of the module.
        """
        super(ExponentialMovingAverage, self).__init__(name=name)
        self._decay = decay
        self._counter = tf.Variable(0, trainable=False, dtype=tf.int64, name='counter')

        self._hidden = None
        self.average = None
        self.initialized = tf.Variable(False, dtype=tf.bool, trainable=False, name='initialized')

    def call(self, value, training=None, **kwargs):
        """Updates the metric and returns the new value."""
        tf.assert_equal(self.initialized, True, message='EMA was not initialized, call initialize() first')
        self.update(value)
        return self.value

    def update(self, value: tf.Tensor):
        """Applies EMA to the value given."""

        self._counter.assign_add(1)
        value = tf.convert_to_tensor(value)
        counter = tf.cast(self._counter, value.dtype)
        self._hidden.assign_sub((self._hidden - value) * (1 - self._decay))
        self.average.assign((self._hidden / (1. - tf.pow(self._decay, counter))))

    @property
    def value(self) -> tf.Tensor:
        """Returns the current EMA."""
        return self.average.read_value()

    def reset(self):
        """Resets the EMA."""
        self._counter.assign(tf.zeros_like(self._counter))
        self._hidden.assign(tf.zeros_like(self._hidden))
        self.average.assign(tf.zeros_like(self.average))

    def initialize(self, value: tf.Tensor):
        self.initialized.assign(True)
        self._hidden = tf.Variable(tf.zeros_like(value), trainable=False, name="hidden")
        self.average = tf.Variable(tf.zeros_like(value), trainable=False, name="average")


class QuantizationLayerEMA(tfkl.Layer):
    """Sonnet module representing the VQ-VAE layer.

    Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    The difference between VectorQuantizerEMA and VectorQuantizer is that
    this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. For
    most experiments the EMA version trains faster than the non-EMA version.

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.

    The output tensor will have the same shape as the input.

    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.

    Attributes:
      embedding_dim: integer representing the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
      num_embeddings: integer, the number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms (see
        equation 4 in the paper).
      decay: float, decay for the moving averages.
      epsilon: small float constant to avoid numerical instability.
    """

    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 commitment_cost,
                 decay,
                 epsilon=1e-5,
                 dtype=tf.float32,
                 name='vector_quantizer_ema',
                 **kwargs):
        """Initializes a VQ-VAE EMA module.

        Args:
          embedding_dim: integer representing the dimensionality of the tensors in
            the quantized space. Inputs to the modules must be in this format as
            well.
          num_embeddings: integer, the number of vectors in the quantized space.
          commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
          decay: float between 0 and 1, controls the speed of the Exponential Moving
            Averages.
          epsilon: small constant to aid numerical stability, default 1e-5.
          dtype: dtype for the embeddings variable, defaults to tf.float32.
          name: name of the module.
        """
        super(QuantizationLayerEMA, self).__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        if not 0 <= decay <= 1:
            raise ValueError('decay must be in range [0, 1]')
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        embedding_shape = [embedding_dim, num_embeddings]
        initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        self.embeddings = tf.Variable(initializer(embedding_shape, dtype), name='embeddings', trainable=False)

        self.ema_cluster_size = ExponentialMovingAverage(decay=self.decay, name='ema_cluster_size')
        self.ema_cluster_size.initialize(tf.zeros([num_embeddings], dtype=dtype))

        self.ema_dw = ExponentialMovingAverage(decay=self.decay, name='ema_dw')
        self.ema_dw.initialize(self.embeddings)

    def call(self, inputs, training=None, **kwargs):
        """Connects the module to some inputs.

        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be flattened and treated as a large batch.
          training: boolean, whether this connection is to training data. When
            this is set to False, the internal moving average statistics will not be
            updated.

        Returns:
          dict containing the following keys and values:
            quantize: Tensor containing the quantized version of the input.
            latent_loss: Tensor containing the latent_loss to optimize.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
            of the quantized space each input element was mapped to.
            encoding_indices: Tensor containing the discrete encoding indices, ie
            which element of the quantized space each input element was mapped to.
        """
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (
                tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
                2 * tf.matmul(flat_inputs, self.embeddings) +
                tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        # NB: if your code crashes with a reshape error on the line below about a
        # Tensor containing the wrong number of values, then the most likely cause
        # is that the input passed in does not have a final dimension equal to
        # self.embedding_dim. Ideally we would catch this with an Assert but that
        # creates various other problems related to device placement / TPUs.
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])

        quantized = self.codebook_lookup(encoding_indices)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)

        if training:
            updated_ema_cluster_size = self.ema_cluster_size(tf.reduce_sum(encodings, axis=0))

            flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])
            dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
            updated_ema_dw = self.ema_dw(dw)

            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)

            normalised_updated_ema_w = (updated_ema_dw / tf.reshape(updated_ema_cluster_size, [1, -1]))

            self.embeddings.assign(normalised_updated_ema_w)
            latent_loss = self.commitment_cost * e_latent_loss
        else:
            latent_loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        #return {
        #    'quantize': quantized,
        #    'latent_loss': latent_loss,
        #    'perplexity': perplexity,
        #    'encodings': encodings,
        #    'encoding_indices': encoding_indices,
        #    'distances': distances,
        #}

        self.add_loss(latent_loss)
        self.add_metric(latent_loss, 'latent_loss')
        self.add_metric(perplexity, 'perplexity')
        self.add_metric(distances, 'distances')

        return quantized

    def encode_to_indices(self, inputs):
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (
                tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
                2 * tf.matmul(flat_inputs, self.embeddings) +
                tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1, output_type=tf.int32)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        return encoding_indices

    def encode_to_embedding_vectors(self, inputs):
        encoding_indices = self.encode_to_indices(inputs)
        quantized = self.codebook_lookup(encoding_indices)
        return quantized

    def codebook_lookup(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(self.embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)

    @tf.function
    def codebook_lookup_straight_through(self, encoding_indices_one_hot):
        """Takes almost one-hot encoded float indices and returns embedding vectors, provides straight-through gradient
        to circumvent argmax for lookup."""

        tf.assert_less(tf.rank(encoding_indices_one_hot), 6, (f'Encoding index tensor should have rank 4 of 5, but '
                                                              f'has {tf.rank(encoding_indices_one_hot)}'))

        s_in = tf.shape(encoding_indices_one_hot)
        s_soft_embeddings = tf.concat([s_in[:-1], tf.expand_dims(self.embedding_dim, 0)], axis=0)
        w = tf.transpose(self.embeddings, [1, 0])  # matrix with embedding vectors as row vectors

        hard_embeddings = tf.nn.embedding_lookup(w, tf.argmax(encoding_indices_one_hot, -1))
        encoding_indices_flat = tf.reshape(encoding_indices_one_hot, (-1, self.num_embeddings))
        soft_embeddings_flat = tf.matmul(encoding_indices_flat, w)
        soft_embeddings = tf.reshape(soft_embeddings_flat, s_soft_embeddings)

        return hard_embeddings + soft_embeddings - tf.stop_gradient(soft_embeddings)


class ResidualStackLayer(tfkl.Layer):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='residual_stack_layer', **kwargs):
        super(ResidualStackLayer, self).__init__(name=name, **kwargs)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = tfkl.Conv2D(
                filters=num_residual_hiddens,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                name="res3x3_%d" % i)
            conv1 = tfkl.Conv2D(
                filters=num_hiddens,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def call(self, inputs, training=None, **kwargs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


class Encoder(tfkl.Layer):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = tfkl.Conv2D(
            filters=self._num_hiddens // 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name="enc_1")
        self._enc_2 = tfkl.Conv2D(
            filters=self._num_hiddens,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name="enc_2")
        self._enc_3 = tfkl.Conv2D(
            filters=self._num_hiddens,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name="enc_3")
        self._residual_stack = ResidualStackLayer(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def call(self, x, training=None, **kwargs):
        h = tf.nn.relu(self._enc_1(x))
        h = tf.nn.relu(self._enc_2(h))
        h = tf.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


class Decoder(tfkl.Layer):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, grayscale_out=False, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = tfkl.Conv2D(
            filters=self._num_hiddens,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name="dec_1")
        self._residual_stack = ResidualStackLayer(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = tfkl.Conv2DTranspose(
            filters=self._num_hiddens // 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name="dec_2")
        self._dec_3 = tfkl.Conv2DTranspose(
            filters=1 if grayscale_out else 3,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name="dec_3")

    def call_with_reshape(self, x, training=None, **kwargs):
        s_in = x.get_shape().as_list()

        tf.assert_less(len(s_in), 6), f'Decoder can at process at most 2 batch dimensions (5D tensor), got {len(s_in)}'
        tf.assert_greater(len(s_in), 3), f'Decoder needs at least 4D tensor (b, w, h, c), but got {len(s_in)}'

        x_flat = tf.reshape(x, (-1, s_in[-3], s_in[-2], s_in[-1]))  # conv2DTranspose must have 4D input

        h = self._dec_1(x_flat)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)

        s_rec = x_recon.get_shape().as_list()
        if len(s_in) == 5:
            x_recon = tf.reshape(x_recon, (-1, s_in[1], s_rec[-3], s_rec[-2], s_rec[-1]))
        return x_recon

    def call(self, x, training=None, **kwargs):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon


class VectorQuantizerEMAKeras(tf.keras.Model):

    def __init__(self, train_data_variance,
                 decay=0.99,
                 commitment_cost=0.25,
                 embedding_dim=64,
                 num_embeddings=512,
                 num_hiddens=128,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 grayscale_input=False):

        super(VectorQuantizerEMAKeras, self).__init__()

        self._encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
        self._decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, grayscale_out=grayscale_input)
        self._vqvae = QuantizationLayerEMA(embedding_dim=embedding_dim, num_embeddings=num_embeddings,
                                           commitment_cost=commitment_cost, decay=decay)
        self._pre_vq_conv1 = tfkl.Conv2D(embedding_dim, kernel_size=(1, 1), strides=(1, 1), name='to_vq')
        self._data_variance = train_data_variance

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.frame_stack = 1
        self._grayscale_input = grayscale_input
        self._total_loss = tf.keras.metrics.Mean('total_loss')
        self._reconstruction_loss = tf.keras.metrics.Mean('reconstruction_loss')


    def call(self, inputs, training=None, **kwargs):
        inputs, timesteps = maybe_flatten_time(inputs, 3)

        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, training=training)
        x_recon = self._decoder(vq_output)

        x_recon = maybe_unflatten_time(x_recon, timesteps)
        #recon_error = tf.reduce_mean((x_recon - inputs) ** 2) / self._data_variance
        #self.add_metric(recon_error, 'reconstruction_loss')

        # use the internal losses implemented by the vq_vae model of sonnet
        #self.add_loss(recon_error)  # reconstruction loss
        #self.add_metric(recon_error, 'reconstruction_loss')
        #self.add_loss(vq_loss)
        #self.add_loss(vq_output['loss'])  # latent loss
        #self.add_metric(recon_error + vq_output['loss'], name='loss')

        return x_recon

    def compute_latent_shape(self, input_shape):
        index_mat = self.encode_to_indices(tf.zeros((1, *input_shape), dtype=tf.float32))
        return index_mat.get_shape()[1:]

    def encode_to_vectors(self, inputs):
        inputs, timesteps = maybe_flatten_time(inputs, 3)

        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae.encode_to_embedding_vectors(z)

        vq_output = maybe_unflatten_time(vq_output, timesteps)
        return vq_output

    def encode_to_indices(self, inputs):
        if isinstance(inputs, tf.data.Dataset):
            vq_outputs_list = []
            for batch in inputs:
                batch, timesteps = maybe_flatten_time(batch, 3)
                z = self._pre_vq_conv1(self._encoder(batch))
                vq_output_batch = self._vqvae.encode_to_indices(z)
                vq_output_batch = maybe_unflatten_time(vq_output_batch, timesteps)
                vq_outputs_list.append(vq_output_batch)
            vq_output = tf.concat(vq_outputs_list, axis=0)
        else:
            inputs, timesteps = maybe_flatten_time(inputs, 3)
            z = self._pre_vq_conv1(self._encoder(inputs))
            vq_output = self._vqvae.encode_to_indices(z)
            vq_output = maybe_unflatten_time(vq_output, timesteps)
        return vq_output

    def encode_to_indices_onehot(self, inputs):
        if isinstance(inputs, tf.data.Dataset):
            vq_outputs_list = []
            for batch in inputs:
                batch, timesteps = maybe_flatten_time(batch, 3)
                z = self._pre_vq_conv1(self._encoder(batch))
                embed_indices = self._vqvae.encode_to_indices(z)
                embed_indices_onehot = tf.one_hot(embed_indices, self.num_embeddings, axis=-1)
                embed_indices_onehot = maybe_unflatten_time(embed_indices_onehot, timesteps)
                vq_outputs_list.append(embed_indices_onehot)
            vq_output = tf.concat(vq_outputs_list, axis=0)
        else:
            inputs, timesteps = maybe_flatten_time(inputs, 3)
            z = self._pre_vq_conv1(self._encoder(inputs))
            embed_indices = self._vqvae.encode_to_indices(z)
            vq_output = tf.one_hot(embed_indices, self.num_embeddings, axis=-1)
            vq_output = maybe_unflatten_time(vq_output, timesteps)
        return vq_output

    def decode_from_vectors(self, embeddings):
        embeddings, timesteps = maybe_flatten_time(embeddings, 3)
        x_recon = self._decoder(embeddings)
        x_recon = maybe_unflatten_time(x_recon, timesteps)
        return x_recon

    def decode_from_indices(self, indices):
        indices, timesteps = maybe_flatten_time(indices, 2)
        embeddings = self._vqvae.codebook_lookup(indices)
        x_recon = self._decoder(embeddings)
        x_recon = maybe_unflatten_time(x_recon, timesteps)
        return x_recon

    def indices_to_embeddings(self, indices):
        embeddings = self._vqvae.codebook_lookup(indices)
        return embeddings

    def indices_to_embeddings_straight_through(self, indices):
        embeddings = self._vqvae.codebook_lookup_straight_through(indices)
        return embeddings

    @tf.function
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_recon = self(x, training=True)
            recon_error = tf.reduce_mean((x_recon - x) ** 2) / self._data_variance
            #loss = self.compiled_loss(x, x_recon, regularization_losses=self.losses)
            #loss += recon_error
            loss = recon_error + self.losses

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self._total_loss.update_state(loss)
        self._reconstruction_loss.update_state(recon_error)
        self.compiled_metrics.update_state(x, x_recon)

        return {m.name: m.result() for m in self.metrics}
