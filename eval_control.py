from replay_memory_tools import *
from tools import *
from project_init import *
from control import control
import neptune.new as neptune
import gc

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

    if CONFIG.neptune_project_name:
        run = neptune.init(project=CONFIG.neptune_project_name)
        run['parameters'] = {k: v for k,v in vars(CONFIG).items()}
        run['sys/tags'].add('control')
        if not CONFIG.tf_eager_mode:
            run['predictor_params'] = pred.count_params()
            run['vae_params'] = vae.count_params()
    else:
        run = None

    for i_run in range(CONFIG.ctrl_n_runs):
        for name, env in zip(env_names, envs):
            print(f'Planning in {name}')
            if name == 'Gridworld-partial-room-v7':
                env.player_random_start = False
            r, t = control(predictor=pred, vae=vae, env=env, env_info=env_info, env_name=name,
                           plan_steps=CONFIG.ctrl_n_plan_steps, warmup_steps=CONFIG.ctrl_n_warmup_steps,
                           n_rollouts=CONFIG.ctrl_n_rollouts, n_iterations=CONFIG.ctrl_n_iterations,
                           top_perc=CONFIG.ctrl_top_perc, gamma=CONFIG.ctrl_gamma,
                           consecutive_actions=CONFIG.ctrl_consecutive_actions,
                           max_steps=CONFIG.ctrl_max_steps, render=CONFIG.ctrl_render, neptune_run=run)

            tf.keras.backend.clear_session()
            gc.collect()

            if run:
                run[f'{name}/rewards'].log(r)
                run[f'{name}/steps'].log(t)

