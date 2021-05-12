from project_init import *
from tools import *
from simulated_env import *

if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    predictor_weights_path = gen_predictor_weights_path(env_names)

    if CONFIG.env_setting == 'gridworld_3_rooms_rand_starts':
        print('Deactivating random starts for control')
        for env in envs:
            env.player_random_start = False

    # load and prepare data
    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)
    del mix_memory

    # instantiate vae and load trained weights
    vae = vq_vae_net(obs_shape=env_info['obs_shape'],
                     n_embeddings=CONFIG.vae_n_embeddings,
                     d_embeddings=CONFIG.vae_d_embeddings,
                     train_data_var=train_data_var,
                     commitment_cost=CONFIG.vae_commitment_cost,
                     frame_stack=CONFIG.vae_frame_stack,
                     summary=CONFIG.model_summaries,
                     tf_eager_mode=CONFIG.tf_eager_mode)
    load_vae_weights(vae=vae, weights_path=vae_weights_path)

    # instantiate predictor
    pred = predictor_net(n_actions=env_info['n_actions'],
                         obs_shape=env_info['obs_shape'],
                         n_envs=len(envs),
                         vae=vae,
                         det_filters=CONFIG.pred_det_filters,
                         prob_filters=CONFIG.pred_prob_filters,
                         decider_lw=CONFIG.pred_decider_lw,
                         n_models=CONFIG.pred_n_models,
                         tensorboard_log=CONFIG.pred_tb_log,
                         summary=CONFIG.model_summaries,
                         tf_eager_mode=CONFIG.tf_eager_mode)
    pred.load_weights(predictor_weights_path)

    #dream_env = MultiSimulatedLatentSpaceEnv(envs, pred, vae, [0, 1, 2], 0.9)
    dream_env = MultiLatentSpaceEnv(envs, vae, [0, 1, 2])
    #dream_env = MultiEnv(envs, [0, 1, 2])
    #dream_env = SimulatedLatentSpaceEnv(envs[2], pred, vae, 0)
    #dream_env = LatentSpaceEnv(envs[2], vae, 0)
    print(dream_env.reset())

    rewards = []
    for t in range(1000):
        dream_env.render()
        a = dream_env.action_space.sample()
        s, r, done, info = dream_env.step(a)
        rewards.append(r)
        if done:
            break
    print(f'Return: {np.sum(rewards)}')