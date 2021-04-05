import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tf_tools import *
from tensorflow.keras import backend as Kb

# Hyperparameters
NUM_LATENT_K = 20                 # Number of codebook entries
NUM_LATENT_D = 64                 # Dimension of each codebook entries
BETA = 1.0                        # Weight for the commitment loss

VQVAE_BATCH_SIZE = 128            # Batch size for training the VQVAE
VQVAE_NUM_EPOCHS = 20             # Number of epochs
VQVAE_LEARNING_RATE = 3e-4        # Learning rate
VQVAE_LAYERS = [16, 32]           # Number of filters for each layer in the encoder


class VectorQuantizer(K.layers.Layer):

    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k
        self.d = None
        self.codebook = None

    def build(self, input_shape):
        self.d = int(input_shape[-1])
        rand_init = K.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(shape=(self.k, self.d), initializer=rand_init, trainable=True, name='codebook')

    def call(self, inputs, **kwargs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1)
        return tf.cast(k_index, tf.int32)

    def sample(self, k_index):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        k_index_one_hot = tf.one_hot(k_index, self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q

    def sample_constant_codebook_onehot(self, k_index_one_hot):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = tf.reshape(tf.stop_gradient(self.codebook), shape=(1, 1, 1, self.k, self.d))
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q


def encoder_pass(inputs, d, filters_per_layer):
    x = inputs
    for i, filters in enumerate(filters_per_layer):
        x = K.layers.Conv2D(filters=filters, kernel_size=5, padding='SAME', activation='relu',
                            strides=(2, 2), name="conv{}".format(i + 1))(x)
        #x = ResidualConv2D(filters, (2, 2), name=f'conv{i + 1}')(x)
    z_e = K.layers.Conv2D(filters=d, kernel_size=5, padding='SAME', activation=None,
                          strides=(1, 1), name='z_e')(x)
    return z_e


def decoder_pass(inputs, input_shape, filters_per_layer):
    y = inputs
    #y = ResidualConv2D(64)(y)
    for i, filters in enumerate(filters_per_layer):
        y = K.layers.Conv2DTranspose(filters=filters, kernel_size=5, strides=(2, 2), padding="SAME",
                                     activation='relu', name="convT{}".format(i + 1))(y)
        y = K.layers.BatchNormalization()(y)
    decoded = K.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(1, 1),
                                       padding="SAME", activation='sigmoid', name='output')(y)
    return decoded


def build_vqvae(k, d, input_shape=(28, 28, 1), num_layers=None):
    global BETA
    num_layers = num_layers or [16, 32, 64]

    ## Encoder
    encoder_inputs = K.layers.Input(shape=input_shape, name='encoder_inputs')
    z_e = encoder_pass(encoder_inputs, d, filters_per_layer=num_layers)
    size = int(z_e.get_shape()[1])

    ## Vector Quantization
    vector_quantizer = VectorQuantizer(k, name="vector_quantizer")
    codebook_indices = vector_quantizer(z_e)
    encoder = K.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

    ## Decoder
    decoder_inputs = K.layers.Input(shape=(size, size, d), name='decoder_inputs')
    decoded = decoder_pass(decoder_inputs, input_shape, filters_per_layer=num_layers[::-1])
    decoder = K.Model(inputs=decoder_inputs, outputs=decoded, name='decoder')

    ## VQVAE Model (training)
    sampling_layer = K.layers.Lambda(lambda x: vector_quantizer.sample(x), name="sample_from_codebook")
    z_q = sampling_layer(codebook_indices)
    codes = tf.stack([z_e, z_q], axis=-1)
    codes = K.layers.Lambda(lambda x: x, name='latent_codes')(codes)
    straight_through = K.layers.Lambda(lambda x: x[1] + tf.stop_gradient(x[0] - x[1]),
                                       name="straight_through_estimator")
    straight_through_zq = straight_through([z_q, z_e])
    reconstructed = decoder(straight_through_zq)
    vq_vae = K.Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')
    #vq_vae = K.Model(inputs=encoder_inputs, outputs=reconstructed, name='vq-vae')

    # add latent loss for regularization
    #vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
    #commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
    #latent_loss = tf.identity(vq_loss + BETA * commit_loss, name="latent_loss")
    #vq_vae.add_loss(latent_loss)

    ## VQVAE model (inference)
    codebook_indices = K.layers.Input(shape=(size, size), name='discrete_codes', dtype='int32')
    sampling_layer = K.layers.Lambda(lambda x: vector_quantizer.sample(x), name="sample_from_codebook")
    z_q = sampling_layer(codebook_indices)
    generated = decoder(z_q)
    vq_vae_sampler = K.Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')

    ## Transition from codebook indices to model (for training the prior later)
    indices = K.layers.Input(shape=(size, size), name='codes_sampler_inputs', dtype='int32')
    z_q = sampling_layer(indices)
    codes_sampler = K.Model(inputs=indices, outputs=z_q, name="codes_sampler")

    ## Transition from codebook indices to model with constant codebook and onehot indices
    one_hot_indices = K.layers.Input(shape=(size, size, k), name='codes_sampler_one_hot_inputs', dtype='float32')
    sampling_layer_constant_codebook = K.layers.Lambda(lambda x: vector_quantizer.sample_constant_codebook_onehot(x), name="sample_from_constant_codebook")
    z_q = sampling_layer_constant_codebook(one_hot_indices)
    codes_sampler_onehot = K.Model(inputs=one_hot_indices, outputs=z_q, name="codes_sampler")


    ## Getter to easily access the codebook for vizualisation
    indices = K.layers.Input(shape=(), dtype='int32')
    vector_model = K.Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]), name='get_codebook')

    def get_vq_vae_codebook():
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook

    return vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, codes_sampler_onehot, get_vq_vae_codebook


def mse_loss(ground_truth, predictions):
    mse_loss = tf.reduce_mean((ground_truth - predictions)**2, name="mse_loss")
    return mse_loss


def latent_loss(dummy_ground_truth, outputs):
    global BETA
    del dummy_ground_truth
    z_e, z_q = tf.split(outputs, 2, axis=-1)
    vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
    commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
    latent_loss = tf.identity(vq_loss + BETA * commit_loss, name="latent_loss")
    return latent_loss

def zq_norm(y_true, y_pred):
    del y_true
    _, z_q = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.norm(z_q, axis=-1))

def ze_norm(y_true, y_pred):
    del y_true
    z_e, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.norm(z_e, axis=-1))


class VqVAE:

    def __init__(self, in_shape, num_latent_k, latent_features, frame_stack=1, beta=1, kl_loss_factor_mul=1.000001):
        if frame_stack > 1:
            in_shape = (in_shape[0], in_shape[1], frame_stack * in_shape[2])

        vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, codes_sampler_onehot, get_vq_vae_codebook = build_vqvae(
            num_latent_k, latent_features, input_shape=in_shape, num_layers=VQVAE_LAYERS)
        #vq_vae.compile(loss=[mse_loss, latent_loss], metrics={"latent_codes": [zq_norm, ze_norm]},
        #               optimizer=K.optimizers.Adam(VQVAE_LEARNING_RATE))

        self.in_shape = in_shape
        self.frame_stack = frame_stack
        self.vq_vae = vq_vae
        self.vq_vae_sampler = vq_vae_sampler
        self.encoder = encoder
        self.decoder = decoder
        self.codes_sampler = codes_sampler
        self.codes_sampler_onehot = codes_sampler_onehot
        self.get_vq_vae_codebook = get_vq_vae_codebook
        self.enc_out_shape = tuple(encoder.compute_output_shape((None, *in_shape))[1:])

        dummy = np.zeros((1, *self.enc_out_shape))
        self.latent_shape = codes_sampler.predict(dummy).shape[1:]
        self.n_cb_vectors = num_latent_k
        self.d_cb_vectors = latent_features
        #self.latent_shape = codes_sampler.compute_output_shape(self.encoder.compute_output_shape(in_shape))[1:]
        #self.latent_shape = (NUM_LATENT_K, latent_features)

    def compile(self, **kwargs):
        metrics = kwargs.pop('metrics', {})
        if type(metrics) is list:
            metrics.append([zq_norm, ze_norm])
        else:
            metrics['latent_codes'] = [zq_norm, ze_norm]
        kwargs['metrics'] = metrics
        kwargs['loss'] = [mse_loss, latent_loss]
        #kwargs['loss'] = [mse_loss]
        #kwargs['run_eagerly'] = True
        self.vq_vae.compile(**kwargs)

    def fit(self, x, **kwargs):
        """
        def _generator(states, next_states, terminal):
            for s, s_, done in zip(states, next_states, terminal):
                yield s.astype(np.float32), s.astype(np.float32) if not done else s_.astype(np.float32), s_.astype(np.float32)

        dataset_generator = tf.data.Dataset.from_generator(
            _generator,
            args=[x['s'], x['s_'], x['done']],
            output_signature=(tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32))
        )
        history = self.vq_vae.fit(dataset_generator, verbose=1, batch_size=batch_size, shuffle=True).history
        """

        #dataset = tf.data.Dataset.from_tensor_slices((x, x))
        #dataset = dataset.shuffle(len(x)).batch(batch_size)
        #history = self.vq_vae.fit(dataset, **kwargs)

        #dataset = tf.data.Dataset.from_tensor_slices((x, [x, np.zeros_like(x)]))
        #dataset = dataset.shuffle(len(dataset)).batch(batch_size)
        #history = self.vq_vae.fit(dataset, *kwargs)

        history = self.vq_vae.fit(x, [x, np.zeros_like(x)], **kwargs)
        #history = self.vq_vae.fit(x, [x, np.zeros_like(x)], **kwargs)
        #if isinstance(x, np.ndarray):
        #    history = self.vq_vae.fit(x, x, *args, **kwargs)
        #else:
        #    history = self.vq_vae.fit(x, *args, **kwargs)

        return history

    def predict(self, x, *args, **kwargs):
        return self.vq_vae.predict(x, *args, **kwargs)[0]
        #return self.vq_vae.predict(x, *args, **kwargs)

    def save_weights(self, path):
        self.vq_vae.save_weights(path)

    def load_weights(self, path):
        self.vq_vae.load_weights(path)

    def encode(self, data, output_type='vectors'):
        #if not isinstance(data, np.ndarray):
        #    data = np.array(data)

        if np.ndim(data) - len(self.in_shape) == 2:
            time_dimension = True
        else:
            time_dimension = False

        batch_size = data.shape[0]
        timesteps = None
        if time_dimension:
            timesteps = data.shape[1]
            data = data.reshape(-1, *self.in_shape)

        # generate codebook indices
        encoded = self.encoder.predict(data)

        # replace each codebook index by its vector if output_type is 'vectors'
        if output_type == 'vectors':
            encoded = self.codes_sampler.predict(encoded)
            if time_dimension:
                encoded = encoded.reshape(batch_size, timesteps, *self.latent_shape)
        elif output_type == 'indices':
            if time_dimension:
                encoded = encoded.reshape(batch_size, timesteps, *self.enc_out_shape)
        else:
            raise ValueError(f'Unknown output type: {output_type}')


        # add last dimension of length 1 to make encoder output a proper image
        #encoded = np.expand_dims(encoded, -1).astype(np.float32)

        return encoded

    def decode(self, data, input_type='vectors'):
        #if not isinstance(data, np.ndarray):
        #    data = np.array(data)

        batch_size = data.shape[0]

        if input_type == 'vectors':
            if np.ndim(data) - len(self.latent_shape) == 2:
                time_dimension = True
                time_dim_folded = (-1, *self.latent_shape)
            else:
                time_dimension = False
                time_dim_folded = None
        elif input_type == 'indices':
            data = data.astype(np.int32)
            if np.ndim(data) - len(self.enc_out_shape) == 2:
                time_dimension = True
                time_dim_folded = (-1, *self.enc_out_shape)
            else:
                time_dimension = False
                time_dim_folded = None
        else:
            raise ValueError(f'Unknown input type: {input_type}')

        if time_dimension:
            timesteps = data.shape[1]
            data = tf.reshape(data, time_dim_folded)
        else:
            timesteps = None

        if input_type == 'vectors':
            decoded = self.decoder(data)
        else:
            decoded = self.vq_vae_sampler(data)

        if time_dimension:
            decoded = tf.reshape(decoded, (batch_size, timesteps, *self.in_shape))

        return decoded
