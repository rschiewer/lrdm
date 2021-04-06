from project_init import *
from tools import gen_environments, vq_vae_net, predictor_net, load_vae_weights,\
    prepare_predictor_data
from replay_memory_tools import load_env_samples, extract_subtrajectories
import numpy as np
import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import pickle
import gc


if __name__ == '__main__':
    # prepare stuff
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    predictor_weights_path = gen_predictor_weights_path(env_names)
    predictor_train_stats_path = gen_predictor_train_stats_path(env_names)

    # load train data and extract trajectories
    mix_memory = load_env_samples(mix_mem_path)
    trajs = extract_subtrajectories(mem=mix_memory,
                                    num_trajectories=CONFIG.pred_n_trajectories,
                                    traj_length=CONFIG.pred_n_traj_steps,
                                    pad_short_trajectories=CONFIG.pred_pad_trajectories)
    train_data_var = np.var(mix_memory['s'][0] / 255)
    del mix_memory  # conserve memory

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

    # rewards = cumulative_episode_rewards(mix_memory)
    # rewards_from_mem = mix_memory['r'].sum()
    # plt.plot(rewards, label='cumulative episode rewards')
    # plt.show()

    # extract trajectories and train predictor
    enc_o, enc_o_, r, a, done = prepare_predictor_data(trajectories=trajs,
                                                       vae=vae,
                                                       n_steps=CONFIG.pred_n_traj_steps,
                                                       n_warmup_steps=CONFIG.pred_n_warmup_steps)

    dataset = (tf.data.Dataset.from_tensor_slices(((enc_o, a), (enc_o_, r, done)))
               .shuffle(100000)
               .repeat(-1)
               .batch(CONFIG.pred_batch_size, drop_remainder=True)
               .prefetch(-1))

    callbacks = []
    if CONFIG.neptune_project_name:
        run = neptune.init(project=CONFIG.neptune_project_name)
        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
        run['parameters'] = {k: v for k,v in vars(CONFIG).items() if k.startswith('pred_')}
        run['sys/tags'].add('predictor')
        callbacks.append(neptune_cbk)

    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            tf.keras.backend.clear_session()
    callbacks.append(MyCustomCallback())

    # all_predictor.load_weights('predictors/' + predictor_weights_path)
    epochs = np.ceil(CONFIG.pred_n_train_steps / CONFIG.pred_n_steps_per_epoch).astype(np.int32)
    history = pred.fit(dataset,
                       epochs=epochs,
                       steps_per_epoch=CONFIG.pred_n_steps_per_epoch,
                       verbose=1,
                       callbacks=callbacks)

    pred.save_weights(predictor_weights_path)
    with open(predictor_train_stats_path, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
