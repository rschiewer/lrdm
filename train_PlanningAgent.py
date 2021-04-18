from project_init import *
from PlanningAgent import *
from tools import gen_environments, ExperimentConfig


if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)

    agent = SplitPlanAgent(obs_shape=env_info['obs_shape'],
                           n_actions=env_info['n_actions'],
                           mem_size=200000,
                           exploration_steps=2000,
                           vae_train_interval=10,
                           pred_train_interval=100,
                           config=CONFIG)
    env_idx = 0
    agent.train(envs[env_idx], env_idx, 200000)

    # store results
    predictor_weights_path = gen_predictor_weights_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    agent.predictor.save_weights(predictor_weights_path)
    agent.vae.save_weights(vae_weights_path)
