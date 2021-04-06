from project_init import *
from tools import gen_environments, vq_vae_net, predictor_net, load_vae_weights, \
    prepare_predictor_data
from replay_memory_tools import load_env_samples, line_up_observations,\
    stack_observations, cast_and_normalize_images, cast_and_unnormalize_images
import numpy as np
import tensorflow as tf
import neptune.new as neptune
import pickle
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    vae_train_stats_path = gen_vae_train_stats_path(env_names)

    mix_memory = load_env_samples(mix_mem_path)
    train_data_var = np.var(mix_memory['s'][0] / 255)

    vae = vq_vae_net(obs_shape=env_info['obs_shape'],
                     n_embeddings=CONFIG.vae_n_embeddings,
                     d_embeddings=CONFIG.vae_d_embeddings,
                     train_data_var=train_data_var,
                     commitment_cost=CONFIG.vae_commitment_cost,
                     frame_stack=CONFIG.vae_frame_stack,
                     summary=CONFIG.model_summaries)

    if vae.frame_stack == 1:
        all_observations = line_up_observations(mix_memory)
    else:
        all_observations = stack_observations(mix_memory, vae.frame_stack)
    print('Total number of training samples: {}'.format(len(all_observations)))

    train_dataset = (tf.data.Dataset.from_tensor_slices(all_observations)
                     .map(cast_and_normalize_images)
                     .shuffle(100000)
                     .repeat(-1)  # repeat indefinitely
                     .batch(CONFIG.vae_batch_size, drop_remainder=True)
                     .prefetch(-1))

    callbacks = []
    if CONFIG.neptune_project_name:
        run = neptune.init(project=CONFIG.neptune_project_name)
        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
        run['parameters'] = {k: v for k,v in vars(CONFIG).items() if k.startswith('vae_')}
        run['sys/tags'].add('vqvae')
        callbacks.append(neptune_cbk)

    epochs = np.ceil(CONFIG.vae_n_train_steps / CONFIG.vae_n_steps_per_epoch).astype(np.int32)
    history = vae.fit(train_dataset,
                      epochs=epochs,
                      steps_per_epoch=CONFIG.vae_n_steps_per_epoch,
                      verbose=1,
                      callbacks=callbacks)

    vae.save_weights(vae_weights_path)
    with open(vae_train_stats_path, 'wb+') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
