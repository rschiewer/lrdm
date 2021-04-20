from replay_memory_tools import *
from tools import *
from project_init import *
from project_init import gen_mix_mem_path, gen_vae_weights_path, gen_predictor_weights_path
import neptune.new as neptune

if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    predictor_weights_path = gen_predictor_weights_path(env_names)

    # load and prepare data
    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)
    mem_sanity_check(mix_memory)

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

    if CONFIG.neptune_project_name:
        run = neptune.init(project=CONFIG.neptune_project_name)
        run['parameters'] = {k: v for k,v in vars(CONFIG).items() if k.startswith('pred_')}
        run['sys/tags'].add('predictor_performance')
        run['predictor_params'] = pred.count_params()
        run['vae_params'] = vae.count_params()
    else:
        run = None

    for i_env, name in enumerate(env_names):
        fig = predictor_allocation_stability(pred, mix_memory, vae, i_env)
        if run:
            run[f'predictor_allocation_{name}'] = neptune.types.File.as_image(fig)

    # some rollout videos
    targets_obs, targets_r, targets_done, o_rollout, r_rollout, done_rollout, w_predictors = \
        generate_test_rollouts(predictor=pred, mem=mix_memory, vae=vae, n_steps=200, n_warmup_steps=5, n_trajectories=4)
    rollout_videos(targets_obs, o_rollout, r_rollout, done_rollout, w_predictors, 'Predictor Test')

    # more thorough rollouts
    targets_obs, targets_r, targets_done, o_rollout, r_rollout, done_rollout, w_predictors =\
        generate_test_rollouts(predictor=pred, mem=mix_memory, vae=vae, n_steps=50, n_warmup_steps=5, n_trajectories=500)

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
    avg_diff_per_timestep = np.mean(np.abs(r_rollout - targets_r), axis=0)
    plt.plot(avg_diff_per_timestep)
    plt.title('Average reward error rollout vs. ground truth')
    plt.show()
    if run:
        for ad in avg_diff_per_timestep:
            run['avg_r_error'].log(ad)

    # terminal probabilities
    avg_diff_per_timestep = np.mean(np.abs(done_rollout - targets_done), axis=0)
    plt.plot(avg_diff_per_timestep)
    plt.title('Average terminal error rollout vs. ground truth')
    plt.show()
    if run:
        for ad in avg_diff_per_timestep:
            run['avg_terminal_error'].log(ad)

    # predictor choice
    fig = plt.figure()
    plt.hist(np.array(w_predictors).flatten())
    plt.show()
    if run:
        run['Predictor choice'] = neptune.types.File.as_image(fig)

    # difference between predicted and true observations
    fig = plt.figure()
    pixel_diff_mean = np.mean(targets_obs - o_rollout, axis=(0, 2, 3, 4))
    pixel_diff_var = np.std(targets_obs - o_rollout, axis=(0, 2, 3, 4))
    x = range(len(pixel_diff_mean))
    plt.plot(x, pixel_diff_mean)
    plt.fill_between(x, pixel_diff_mean - pixel_diff_var, pixel_diff_mean + pixel_diff_var, alpha=0.2)
    plt.show()
    if run:
        for ad in pixel_diff_mean:
            run['avg_obs_error'].log(ad)
        run['Observation deviation vs groundtruth'] = neptune.types.File.as_image(fig)
