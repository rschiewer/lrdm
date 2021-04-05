from keras_vq_vae import VectorQuantizerEMAKeras
from sonnet_vq_vae import *
from replay_memory_tools import *
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


if __name__ == '__main__':
    batch_size = 128
    train_steps = 1000

    # load data and preprocess
    mix_memory = load_env_samples('samples/mix/Boxing-v0_and_SpaceInvaders-v0_and_Enduro-v0_mix')
    train_data_var = np.var(mix_memory['s']) / 255

    def cast_and_normalise_images(images):
        """Convert images to floating point with the range [-0.5, 0.5]"""
        images = (tf.cast(images, tf.float32) / 255.0) - 0.5
        return images

    train_dataset = (tf.data.Dataset.from_tensor_slices(mix_memory['s'])
            .map(cast_and_normalise_images)
            .shuffle(20000)
            .repeat(-1)  # repeat indefinitely
            .batch(batch_size, drop_remainder=True)
            .prefetch(-1))

    #tf.config.run_functions_eagerly(True)
    vae_keras_port = VectorQuantizerEMAKeras(train_data_var, num_embeddings=512)
    vae_keras_port.compile(tf.keras.optimizers.Adam())
    history_keras = vae_keras_port.fit(train_dataset, epochs=1, steps_per_epoch=train_steps, verbose=1).history

    # start training
    vae_sonnet, train_routine_vae_sonnet = build_sonnet_vqvae(train_data_var, num_embeddings=512)
    history_sonnet = train_routine_vae_sonnet(train_dataset, train_steps)

    for k, v in history_sonnet.items():
        plt.plot(v, label='sonnet_' + k, linestyle='--')
    for k, v in history_keras.items():
        plt.plot(v, label='keras_' + k)
    plt.yscale('log')
    plt.legend()
    plt.show()


    # test
    for i in range(10):
        trajectory = extract_subtrajectories(mix_memory, 1, 300)[0]
        observations = cast_and_normalise_images(trajectory['s'])
        reconstructed_sonnet = vae_sonnet(observations, is_training=False)['x_recon']
        reconstructed_keras = vae_keras_port(observations)
        reconstructed_keras_2 = vae_keras_port.decode_from_indices(vae_keras_port.encode_to_indices(observations))

        observations = np.repeat(observations, 3, axis=-1) + 0.5
        reconstructed_keras += 0.5
        reconstructed_keras_2 += 0.5
        reconstructed_sonnet += 0.5

        trajectory_video([observations, reconstructed_keras, reconstructed_keras_2, reconstructed_sonnet], ['true', 'keras', 'keras2', 'sonnet'], max_cols=4)