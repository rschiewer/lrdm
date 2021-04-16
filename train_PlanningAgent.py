from project_init import *
from PlanningAgent import *
from tools import gen_environments


if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    params_vae = ParamsVae(CONFIG.vae_decay,
                           CONFIG.vae_commitment_cost,
                           CONFIG.vae_d_embeddings,
                           CONFIG.vae_n_embeddings,
                           CONFIG.vae_n_hiddens,
                           CONFIG.vae_n_residual_hiddens,
                           CONFIG.vae_n_residual_layers,
                           CONFIG.vae_batch_size)
    params_pred = ParamsPredictor(CONFIG.pred_det_filters,
                                  CONFIG.pred_prob_filters,
                                  CONFIG.pred_n_models,
                                  CONFIG.pred_decider_lw,
                                  CONFIG.pred_batch_size,
                                  CONFIG.pred_n_traj_steps,
                                  CONFIG.pred_n_warmup_steps)
    params_plan = ParamsPlanning(CONFIG.ctrl_n_plan_steps,
                                 CONFIG.ctrl_n_warmup_steps,
                                 CONFIG.ctrl_n_rollouts,
                                 CONFIG.ctrl_n_iterations,
                                 CONFIG.ctrl_top_perc,
                                 CONFIG.ctrl_gamma,
                                 CONFIG.ctrl_do_mpc,
                                 CONFIG.ctrl_max_steps,
                                 CONFIG.ctrl_render)
    params_plan = ParamsPlanning(40,
                                 4,
                                 50,
                                 3,
                                 CONFIG.ctrl_top_perc,
                                 CONFIG.ctrl_gamma,
                                 CONFIG.ctrl_do_mpc,
                                 CONFIG.ctrl_max_steps,
                                 CONFIG.ctrl_render)

    agent = SplitPlanAgent(obs_shape=env_info['obs_shape'],
                           n_actions=env_info['n_actions'],
                           mem_size=200000,
                           warmup_steps=500,
                           params_vae=params_vae,
                           params_predictor=params_pred,
                           params_planning=params_plan)
    agent.train(envs[0], 200000)

    # store results
    predictor_weights_path = gen_predictor_weights_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    agent.predictor.save_weights(predictor_weights_path)
    agent.vae.save_weights(vae_weights_path)
