from sonnet.src.conv import *
from sonnet.src.conv_transpose import *
from sonnet.src.nets.vqvae import VectorQuantizerEMA, VectorQuantizer
from sonnet.src.optimizers.adam import Adam
from datetime import datetime


class ResidualStack(base.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = Conv2D(
                output_channels=num_residual_hiddens,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="res3x3_%d" % i)
            conv1 = Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


class Encoder(base.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = Conv2D(
            output_channels=self._num_hiddens // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")
        self._enc_2 = Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")
        self._enc_3 = Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def __call__(self, x):
        h = tf.nn.relu(self._enc_1(x))
        h = tf.nn.relu(self._enc_2(h))
        h = tf.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


class Decoder(base.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name=None):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = Conv2DTranspose(
            output_channels=self._num_hiddens // 2,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")
        self._dec_3 = Conv2DTranspose(
            output_channels=3,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")

    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon


class VQVAEModel(base.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1,
                 data_variance, name=None):
        super(VQVAEModel, self).__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def __call__(self, inputs, is_training):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, is_training=is_training)
        x_recon = self._decoder(vq_output['quantize'])
        recon_error = tf.reduce_mean((x_recon - inputs) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output': vq_output,
        }


def build_sonnet_vqvae(train_data_variance, num_embeddings=512, num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2):
    # This value is not that important, usually 64 works.
    # This will not change the capacity in the information-bottleneck.
    embedding_dim = 64

    # The higher this value, the higher the capacity in the information bottleneck.
    #num_embeddings = 10

    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost = 0.25

    # This is only used for EMA updates.
    decay = 0.99
    learning_rate = 3e-4

    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    pre_vq_conv1 = Conv2D(output_channels=embedding_dim, kernel_shape=(1, 1), stride=(1, 1), name="to_vq")
    vq_vae = VectorQuantizerEMA(embedding_dim=embedding_dim, num_embeddings=num_embeddings,
                                         commitment_cost=commitment_cost, decay=decay)
    #vq_vae = VectorQuantizer(embedding_dim=embedding_dim, num_embeddings=num_embeddings, commitment_cost=commitment_cost)

    model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1, data_variance=train_data_variance)
    optimizer = Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            model_output = model(data, is_training=True)
        trainable_variables = model.trainable_variables
        grads = tape.gradient(model_output['loss'], trainable_variables)
        optimizer.apply(grads, trainable_variables)

        return model_output

    def train_routine(train_dataset, steps):
        train_losses = []
        train_recon_errors = []
        train_perplexities = []
        train_vqvae_loss = []

        start = datetime.now()

        for step_index, data in enumerate(train_dataset):
            train_results = train_step(data)
            train_losses.append(train_results['loss'].numpy())
            train_recon_errors.append(train_results['recon_error'].numpy())
            train_perplexities.append(train_results['vq_output']['perplexity'].numpy())
            train_vqvae_loss.append(train_results['vq_output']['loss'].numpy())

            if (step_index + 1) % 100 == 0:
                print('%d train loss: %f ' % (step_index + 1, np.mean(train_losses[-100:])) +
                      ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                      ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                      ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
            if step_index == steps:
                break

        finish = datetime.now()

        print(f'Training finished after {(finish - start).seconds} seconds')

        return {'loss': train_losses,
                'reconstruction loss': train_recon_errors,
                'perplexity': train_perplexities,
                'vq-vae loss': train_vqvae_loss}

    return model, train_routine


