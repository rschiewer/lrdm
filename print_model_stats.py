from project_init import *
from tools import gen_environments, vq_vae_net, predictor_net
from project_init import gen_mix_mem_path
from replay_memory_tools import load_env_samples
import numpy as np


if __name__ == '__main__':
    # prepare stuff
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)

    # load train data and extract trajectories
    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)
    del mix_memory  # conserve memory

    # instantiate vae and load trained weights
    vae = vq_vae_net(obs_shape=env_info['obs_shape'],
                     n_embeddings=CONFIG.vae_n_embeddings,
                     d_embeddings=CONFIG.vae_d_embeddings,
                     train_data_var=train_data_var,
                     commitment_cost=CONFIG.vae_commitment_cost,
                     frame_stack=CONFIG.vae_frame_stack,
                     summary=True,
                     tf_eager_mode=CONFIG.tf_eager_mode)

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
                         summary=True,
                         tf_eager_mode=CONFIG.tf_eager_mode)