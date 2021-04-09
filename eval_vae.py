from replay_memory_tools import *
from tools import *
from project_init import *
from project_init import gen_mix_mem_path, gen_vae_weights_path, gen_vae_train_stats_path, gen_predictor_weights_path
from sklearn.manifold import TSNE

if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    vae_train_stats_path = gen_vae_train_stats_path(env_names)
    predictor_weights_path = gen_predictor_weights_path(env_names)

    # load and prepare data
    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)

    # instantiate vae and load trained weights
    vae = vq_vae_net(obs_shape=env_info['obs_shape'],
                     n_embeddings=CONFIG.vae_n_embeddings,
                     d_embeddings=CONFIG.vae_d_embeddings,
                     train_data_var=train_data_var,
                     commitment_cost=CONFIG.vae_commitment_cost,
                     frame_stack=CONFIG.vae_frame_stack,
                     summary=CONFIG.model_summaries)

    load_vae_weights(vae=vae, weights_path=vae_weights_path, train_stats_path=vae_train_stats_path,
                     plot_training=True, test_memory=mix_memory)

    batch_size = 32
    obs_datset = (tf.data.Dataset.from_tensor_slices(mix_memory[::10]['s'])
                  .map(cast_and_normalize_images)
                  .batch(batch_size, drop_remainder=False)
                  .prefetch(-1))
    encoded_obs = vae.encode_to_indices(obs_datset)

    embedded = TSNE(n_components=2).fit_transform(encoded_obs)
    plt.scatter(embedded[:, 0], embedded[:, 1])


