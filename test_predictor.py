from replay_memory_tools import *
from train_tools import *
from project_init import *

if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
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
    load_vae_weights(vae=vae, weights_path=vae_weights_path)

    # instantiate predictor
    pred = predictor_net(n_actions=env_info['n_actions'],
                         obs_shape=env_info['obs_shape'],
                         vae=vae,
                         det_filters=CONFIG.pred_det_filters,
                         prob_filters=CONFIG.pred_prob_filters,
                         decider_lw=CONFIG.pred_decider_lw,
                         n_models=CONFIG.pred_n_models,
                         tensorboard_log=CONFIG.pred_tb_log,
                         summary=CONFIG.model_summaries)
    pred.load_weights(predictor_weights_path)


    targets, o_rollout, r_rollout, done_rollout, w_predictors = generate_test_rollouts(predictor=pred,
                                                                                       mem=mix_memory,
                                                                                       vae=vae,
                                                                                       n_steps=200,
                                                                                       n_warmup_steps=10,
                                                                                       n_trajectories=4)
    rollout_videos(targets, o_rollout, r_rollout, done_rollout, w_predictors, 'Predictor Test')

    # rewards
    for i, r_traj in enumerate(r_rollout):
        plt.plot(np.squeeze(r_traj), label=f'reward rollout {i}')
    plt.legend()
    plt.show()

    # terminal probabilities
    for i, done_traj in enumerate(done_rollout):
        plt.plot(np.squeeze(done_traj), label=f'terminal prob rollout {i}')
    plt.legend()
    plt.show()

    # predictor choice
    plt.hist(np.array(w_predictors).flatten())
    plt.show()

    # difference between predicted and true observations
    pixel_diff_mean = np.mean(targets - o_rollout, axis=(0, 2, 3, 4))
    pixel_diff_var = np.std(targets - o_rollout, axis=(0, 2, 3, 4))
    x = range(len(pixel_diff_mean))
    plt.plot(x, pixel_diff_mean)
    plt.fill_between(x, pixel_diff_mean - pixel_diff_var, pixel_diff_mean + pixel_diff_var, alpha=0.2)
    plt.show()
