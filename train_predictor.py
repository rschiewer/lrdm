from project_init import *
from replay_memory_tools import cast_and_unnormalize_images
from tools import *
from replay_memory_tools import *
import numpy as np
import tensorflow as tf
import neptune.new as neptune
import glob
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from tensorflow.keras.callbacks import ModelCheckpoint
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
    mem_sanity_check(mix_memory)
    indices, occurrences = np.unique(mix_memory['env'], return_counts=True)
    #trajs = extract_subtrajectories(mem=mix_memory,
    #                                num_trajectories=CONFIG.pred_n_trajectories,
    #                                traj_length=CONFIG.pred_n_traj_steps,
    #                                pad_short_trajectories=CONFIG.pred_pad_trajectories)
    trajs = extract_subtrajectories_unbiased(mix_memory, CONFIG.pred_n_trajectories,
                                             CONFIG.pred_n_traj_steps)
    train_data_var = np.var(mix_memory['s'][0] / 255)
    del mix_memory  # conserve memory

    #trajectory_video(trajs[2000:2010]['s_'], '0123456789')

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

    # rewards = cumulative_episode_rewards(mix_memory)
    # rewards_from_mem = mix_memory['r'].sum()
    # plt.plot(rewards, label='cumulative episode rewards')
    # plt.show()

    # extract trajectories and train predictor
    enc_o, enc_o_, r, a, done, i_env = prepare_predictor_data(trajectories=trajs,
                                                              vae=vae,
                                                              n_steps=CONFIG.pred_n_traj_steps,
                                                              n_warmup_steps=CONFIG.pred_n_warmup_steps)

    #dec_o_ = cast_and_unnormalize_images(vae.decode_from_indices(tf.cast(enc_o_, tf.int32)))
    #trajectory_video(np.stack([dec_o_[4000].numpy(), trajs[4000]['s_']]), 'ro')


    if not CONFIG.pred_use_env_idx:
        i_env = np.ones_like(i_env) * -1

    dataset = (tf.data.Dataset.from_tensor_slices(((enc_o, a), (enc_o_, r, done, i_env)))
               .shuffle(500000)
               .repeat(-1)
               .batch(CONFIG.pred_batch_size, drop_remainder=True)
               .prefetch(-1))

    callbacks = []

    checkpoint_cbk = ModelCheckpoint(predictor_weights_path, monitor='loss', verbose=0, period=1,
                                     save_weights_only=True, save_best_only=True)
    #callbacks.append(checkpoint_cbk)

    if CONFIG.neptune_project_name:
        run = neptune.init(project=CONFIG.neptune_project_name)
        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
        run['parameters'] = {k: v for k,v in vars(CONFIG).items() if k.startswith('pred_')}
        run['sys/tags'].add('predictor')
        if not CONFIG.tf_eager_mode:
            run['predictor_params'] = pred.count_params()
        run['vae_params'] = vae.count_params()
        callbacks.append(neptune_cbk)
    else:
        run = None

    #class MyCustomCallback(tf.keras.callbacks.Callback):
    #    def on_epoch_end(self, epoch, logs=None):
    #        gc.collect()
    #        tf.keras.backend.clear_session()
    #callbacks.append(MyCustomCallback())

    #pred.load_weights(predictor_weights_path)
    epochs = np.ceil(CONFIG.pred_n_train_steps / CONFIG.pred_n_steps_per_epoch).astype(np.int32)
    history = pred.fit(dataset,
                       epochs=epochs,
                       batch_size=CONFIG.pred_batch_size,
                       steps_per_epoch=CONFIG.pred_n_steps_per_epoch,
                       verbose=1,
                       callbacks=callbacks)

    pred.save_weights(predictor_weights_path)
    if run:
        for name in glob.glob(f'{predictor_weights_path}*'):
            run[f'model_weights/{name}'].upload(name)

    # custom train loop because of memory leak in train_step, try to fix in the future
    #dset_iter = iter(dataset)
    #for epoch in range(epochs):
    #    for step in range(CONFIG.pred_n_steps_per_epoch):
    #        batch = next(dset_iter)
    #        loss_stats = pred.train_step(batch)
    #        train_str = f'{epoch + 1}/{epochs}: '
    #        for k, v in loss_stats.items():
    #            train_str += f'{k}: {v:3.5f}\t'
    #            if run:
    #                run[k].log(v)
    #        print(train_str)

    #with open(predictor_train_stats_path, 'wb') as handle:
    #    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
