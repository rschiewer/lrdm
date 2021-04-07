from replay_memory_tools import *
from tools import *
from project_init import *
from project_init import gen_mix_mem_path, gen_vae_weights_path, gen_predictor_weights_path

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

    # some rollout videos
    targets_obs, targets_r, targets_done, o_rollout, r_rollout, done_rollout, w_predictors = \
        generate_test_rollouts(predictor=pred, mem=mix_memory, vae=vae, n_steps=200, n_warmup_steps=10, n_trajectories=4)
    #rollout_videos(targets, o_rollout, r_rollout, done_rollout, w_predictors, 'Predictor Test')

    # more thorough rollouts
    targets_obs, targets_r, targets_done, o_rollout, r_rollout, done_rollout, w_predictors =\
        generate_test_rollouts(predictor=pred, mem=mix_memory, vae=vae, n_steps=5, n_warmup_steps=10, n_trajectories=1000)

    print(np.nonzero(targets_r[:, 0] > 0.75))
    print(np.nonzero(targets_r[:, 1] > 0.75))
    print(np.nonzero(targets_r[:, 2] > 0.75))
    print(np.nonzero(targets_r[:, 3] > 0.75))

    plt.plot(np.mean(np.abs(targets_r - r_rollout), axis=0))
    plt.show()

    #terminal_reward_wrt_timestep = np.full((20,), -1, dtype=np.float32)
    #terminals_wrt_timestep = np.zeros((20,))
    #for r, done in zip(r_rollout, done_rollout):
    #    if np.any(done):
    #        t_terminal = np.argwhere(done == True)
    #        terminal_reward_wrt_timestep[t_terminal] += r[t_terminal]
    #        terminals_wrt_timestep[t_terminal] += 1
    #terminal_reward_wrt_timestep /= terminals_wrt_timestep

    #x_vals = [i for i, terminal_transition_seen in enumerate(terminals_wrt_timestep) if terminal_transition_seen > 0]

    #plt.scatter(x_vals, terminal_reward_wrt_timestep[x_vals])
    #plt.show()

    # rewards
    for i, r_traj in enumerate(r_rollout):
        plt.plot(r_traj, label=f'reward rollout {i}')
    plt.legend()
    plt.show()

    # terminal probabilities
    for i, done_traj in enumerate(done_rollout):
        plt.plot(done_traj, label=f'terminal prob rollout {i}')
    plt.legend()
    plt.show()

    # predictor choice
    plt.hist(np.array(w_predictors).flatten())
    plt.show()

    # difference between predicted and true observations
    pixel_diff_mean = np.mean(targets_obs - o_rollout, axis=(0, 2, 3, 4))
    pixel_diff_var = np.std(targets_obs - o_rollout, axis=(0, 2, 3, 4))
    x = range(len(pixel_diff_mean))
    plt.plot(x, pixel_diff_mean)
    plt.fill_between(x, pixel_diff_mean - pixel_diff_var, pixel_diff_mean + pixel_diff_var, alpha=0.2)
    plt.show()
