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
                     summary=CONFIG.model_summaries,
                     tf_eager_mode=CONFIG.tf_eager_mode)

    load_vae_weights(vae=vae, weights_path=vae_weights_path, train_stats_path=vae_train_stats_path,
                     plot_training=False, test_memory=mix_memory)

    # codebook vector occupancy
    obs_datset = (tf.data.Dataset.from_tensor_slices(mix_memory['s'])
                  .map(cast_and_normalize_images)
                  .batch(64, drop_remainder=False)
                  .prefetch(-1))
    encoded_obs = vae.encode_to_indices(obs_datset)
    n_occurrences = np.bincount(encoded_obs.numpy().flatten())
    plt.bar(range(len(n_occurrences)), n_occurrences)
    plt.title('Codebook vector usage')
    plt.show()

    # get reward observations vs. non-rewarding ones
    rewarding_samples = []
    normal_samples = []
    for i, sample in enumerate(mix_memory):
        if sample['r'] > 0:
            rewarding_samples.append(i)
        else:
            normal_samples.append(i)

    rew_trans = mix_memory[rewarding_samples]
    print(rew_trans.shape)
    rew_obs_datset = (tf.data.Dataset.from_tensor_slices(rew_trans['s_'])
                      .map(cast_and_normalize_images)
                      .batch(64, drop_remainder=False)
                      .prefetch(-1))
    rew_encoded_obs = vae.encode_to_indices(rew_obs_datset)
    rew_n_occurrences = np.bincount(rew_encoded_obs.numpy().flatten()) * 10
    plt.bar(range(len(rew_n_occurrences)), rew_n_occurrences, label='rewarding', alpha=0.5)

    norm_trans = mix_memory[normal_samples]
    print(norm_trans.shape)
    norm_obs_datset = (tf.data.Dataset.from_tensor_slices(norm_trans['s'])
                      .map(cast_and_normalize_images)
                      .batch(64, drop_remainder=False)
                      .prefetch(-1))
    norm_encoded_obs = vae.encode_to_indices(norm_obs_datset)
    norm_n_occurrences = np.bincount(norm_encoded_obs.numpy().flatten())
    plt.bar(range(len(norm_n_occurrences)), norm_n_occurrences, label='normal', alpha=0.5)
    plt.legend()

    plt.show()



