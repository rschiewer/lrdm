from old_predictors import AutoregressivePredictor, AutoregressiveMultiHeadPredictor
from vae import *
from vq_vae import *
from tensorflow.keras import layers
import sksfa
from sklearn.preprocessing import PolynomialFeatures
from replay_memory_tools import *
from matplotlib import pyplot as plt

#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

# see https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def pos_finder_net(obs_shape):
    x_in = keras.Input(obs_shape, name='encoder_input')
    x = layers.Conv2D(64, 5, 1, activation='relu')(x_in)
    # x = layers.Conv2D(64, 5, 1, activation='relu')(x)
    x = layers.Conv2D(32, 5, 1, activation='relu')(x)
    x = layers.Conv2D(32, 3, 1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(2, activation=None)(x)

    pos_finder = keras.Model(inputs=x_in, outputs=x)
    pos_finder.compile(loss='mse', optimizer=tf.optimizers.Adam())
    return pos_finder


def train_pos_finder_net(pos_finder, memories, epochs):
    all_observations = np.concatenate([mem['s_'] / 255 for mem in memories], axis=0)
    all_positions = np.concatenate([mem['pos'] for mem in memories], axis=0)
    pos_finder.fit(all_observations, all_positions, batch_size=64, epochs=epochs, shuffle=True, validation_split=0.2)
    pos_finder.save_weights('position_finder_network/weights')


def load_pos_finder_weights(pos_finder, memories, pixel_size):
    pos_finder.load_weights('position_finder_network/weights')

    num_samples = 500
    mem_idx = np.random.randint(0, len(memories))
    idx = np.random.randint(0, len(memories[mem_idx]) - num_samples)
    obs = memories[mem_idx]['s_'][idx:idx+num_samples] / 255
    pos = memories[mem_idx]['pos'][idx:idx+num_samples].astype(int)

    # delete player position
    # obs[:, pos[0] * pixel_size : (pos[0] + 1) * pixel_size, pos[1] * pixel_size: (pos[1] + 1) * pixel_size] = 1
    pred_pos = pos_finder.predict(obs)
    diff = np.abs(pred_pos - pos).mean()

    print(f'Position estimation absolute error for {num_samples} samples: {diff:.5f}')


def vae_net(obs_shape, latent_size):
    vae = LargeBetaVAE(obs_shape, latent_size, beta=1, kl_loss_factor_mul=1.000001)
    vae.compile(optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    return vae


def train_vae(vae, memories, epochs):
    all_observations = np.concatenate([np.append(mem['s'], mem['s_'][np.newaxis, -1], axis=0) for mem in memories], axis=0)
    print('Total number of training samples: {}'.format(len(all_observations)))
    all_observations = (all_observations / 255.0).astype(np.float32)
    #enc = vae.encode(all_observations[0:1])
    #print(all_observations[0:1].shape)
    #print(np.shape(enc))
    #quit()
    history = vae.fit(all_observations, epochs=epochs, verbose=1, batch_size=256, shuffle=True, validation_split=0.1).history

    vae.save_weights('vae_model/weights')
    with open('vae_model/train_stats', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_vae_weights(vae, memories, plots=False):
    vae.load_weights('vae_model/weights')
    vae.compile(optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    with open('vae_model/train_stats', 'rb') as handle:
        history = pickle.load(handle)

    if plots:
        for stat_name, stat_val in history.items():
            plt.plot(stat_val, label=stat_name)

        plt.title('VAE train stats')
        plt.legend()
        plt.show()

        traj = None
        while traj is None:
            try:
                mem_idx = np.random.randint(0, len(memories))
                traj = extract_subtrajectories(memories[mem_idx], 1, 100, False)[0]
            except ValueError:
                traj = None

        reconstructed = np.clip(vae.predict(traj['s'] / 255), 0, 1)

        anim = trajectory_video([traj['s'] / 255, reconstructed], ['true', 'reconstructed'])
        plt.show()


def test_vae(vae, memories):
    all_observations = np.concatenate([np.append(mem['s'], mem['s_'][np.newaxis, -1], axis=0) for mem in memories], axis=0) / 255
    reconstructed = np.clip(vae.predict(all_observations), 0, 1)

    diff = np.abs(all_observations - reconstructed)
    plt.plot(diff.mean(axis=(1, 2, 3)))
    plt.show()


def predictor_nets_2(latent_size, num_actions, num_heads):
    pre_lws = [[32], [32]]
    intermediate_lws = [32, 32]
    lstm_lw = 64
    post_lws = [16]
    input_shape = (latent_size, num_actions)

    multihead_predictor = AutoregressiveMultiHeadPredictor(input_shape, latent_size, pre_lws=pre_lws, intermediate_lws=intermediate_lws, post_lws=post_lws, lstm_lw=lstm_lw, num_heads=num_heads)

    # all-predictor needs roughly triple the neurons for fairness
    #pre_lws = [[lw * 3 for lw in branch_lws] for branch_lws in pre_lws]
    #intermediate_lws = [lw * 3 for lw in intermediate_lws]
    post_lws = [lw * 3 for lw in post_lws]
    #lstm_lw *= 3
    all_predictor = AutoregressiveMultiHeadPredictor(input_shape, latent_size, pre_lws=pre_lws, intermediate_lws=intermediate_lws, post_lws=post_lws, lstm_lw=lstm_lw, num_heads=1)
    complete_predictors_list = [multihead_predictor, all_predictor]

    for pred in complete_predictors_list:
        pred.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])#, run_eagerly=True)

    return complete_predictors_list


def predictor_nets(latent_size, num_actions, env_names):
    pre_lws = [[32], [32]]
    intermediate_lws = [32]
    post_lws = [32]
    lstm_lw = 32
    input_shape = (latent_size, num_actions)
    predictors = [AutoregressiveMultiHeadPredictor(input_shape, latent_size, pre_lws=pre_lws, intermediate_lws=intermediate_lws, post_lws=post_lws, lstm_lw=lstm_lw, num_heads=1)
                  for name in env_names]
    # all-predictor needs roughly triple the neurons for fairness
    pre_lws = [[lw * 3 for lw in branch_lws] for branch_lws in pre_lws]
    intermediate_lws = [lw * 3 for lw in intermediate_lws]
    post_lws = [lw * 3 for lw in post_lws]
    lstm_lw *= 3
    all_predictor = AutoregressiveMultiHeadPredictor(input_shape, latent_size, pre_lws=pre_lws, intermediate_lws=intermediate_lws, post_lws=post_lws, lstm_lw=lstm_lw, num_heads=1)
    complete_predictors_list = predictors + [all_predictor]

    for pred in complete_predictors_list:
        pred.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])#, run_eagerly=True)

    return complete_predictors_list


def train_predictor_2(vae, predictor, n_actions, trajectories, env_idx, epochs, n_steps, n_warmup_steps=1):
    obs = (trajectories['s'] / 255).astype(np.float32)
    next_obs = (trajectories['s_'] / 255).astype(np.float32)
    one_hot_acts = tf.one_hot(trajectories['a'], n_actions).numpy()

    #assert all([s == env_idx[0] for s in env_idx])

    encoded_obs = vae.encode(obs)
    encoded_next_obs = vae.encode(next_obs)
    encoded_start_obs = encoded_obs[:, 0:n_warmup_steps]
    encoded_targets = encoded_next_obs[:, :n_steps]
    action_inputs = one_hot_acts[:, 0:n_steps]

    predictor.predict_steps = n_steps

    h = predictor.fit([env_idx, encoded_start_obs, action_inputs],
                      encoded_targets,
                      epochs=epochs,
                      batch_size=32,
                      shuffle=True,
                      verbose=1,
                      validation_split=0.10)

    return h.history


def train_predictor(vae, predictor, n_actions, trajectories, epochs, n_steps, n_warmup_steps=1):
    obs = (trajectories['s'] / 255).astype(np.float32)
    next_obs = (trajectories['s_'] / 255).astype(np.float32)
    one_hot_acts = tf.one_hot(trajectories['a'], n_actions).numpy()

    encoded_obs = vae.encode(obs)
    encoded_next_obs = vae.encode(next_obs)
    encoded_start_obs = encoded_obs[:, 0:n_warmup_steps]
    encoded_targets = encoded_next_obs[:, :n_steps]
    action_inputs = one_hot_acts[:, 0:n_steps]

    #anim = trajectory_video(obs[0: 20], 'asdf')
    #plt.show()

    predictor.predict_steps = n_steps

    h = predictor.fit([encoded_start_obs, action_inputs],
                      encoded_targets,
                      epochs=epochs,
                      batch_size=64,
                      shuffle=True,
                      verbose=1,
                      validation_split=0.10)

    return h.history


def train_predictors_2(complete_predictors_list, vae, memories, simulate_agent_memories, n_actions, n_epochs, n_subtrajectories, n_steps, n_warmup_steps):
    multihead_predictor = complete_predictors_list[0]
    all_predictor = complete_predictors_list[1]

    histories = [(n_epochs, n_subtrajectories, n_steps, n_warmup_steps)]
    for i, mem in enumerate(simulate_agent_memories):
        print(f'Training multihead predictor\'s head {i + 1} on memory mix {i + 1} with time window {n_steps}, warmup {n_warmup_steps} and {n_epochs} epochs')
        trajs = extract_subtrajectories(mem, n_subtrajectories, n_steps + n_warmup_steps, False)
        env_idx = trajs['env'][:, 0]
        h = train_predictor_2(vae, multihead_predictor, n_actions, trajs, env_idx, n_epochs, n_steps, n_warmup_steps)
        histories.append(h)
    for i, mem in enumerate(simulate_agent_memories):
        print(f'Training all-predictor on memory mix {i + 1} with time window {n_steps}, warmup {n_warmup_steps} and {n_epochs} epochs')
        trajs = extract_subtrajectories(mem, n_subtrajectories, n_steps + n_warmup_steps, False)
        env_idx = np.zeros_like(trajs['env'][:, 0])
        h = train_predictor_2(vae, all_predictor, n_actions, trajs, env_idx, n_epochs, n_steps, n_warmup_steps)
        histories.append(h)

    for i, pred in enumerate(complete_predictors_list):
        pred.save_weights(f'predictors/predictor_{i}')
    with open('predictors/train_stats', 'wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_predictors(complete_predictors_list, vae, memories, simulate_agent_memories, n_actions, n_epochs, n_subtrajectories, n_steps, n_warmup_steps):
    predictors = complete_predictors_list[0:3]
    all_predictor = complete_predictors_list[3]

    histories = [(n_epochs, n_subtrajectories, n_steps, n_warmup_steps)]
    for i, (mem, pred) in enumerate(zip(memories, predictors)):
        print(f'Training predictor on memory {i + 1} with time window {n_steps}, warmup {n_warmup_steps} and {n_epochs} epochs')
        trajs = extract_subtrajectories(mem, n_subtrajectories, n_steps + n_warmup_steps, False)
        h = train_predictor(vae, pred, n_actions, trajs, n_epochs, n_steps, n_warmup_steps)
        histories.append(h)
    for i, mem in enumerate(simulate_agent_memories):
        print(f'Training all-predictor on memory {i + 1} with time window {n_steps}, warmup {n_warmup_steps} and {n_epochs} epochs')
        trajs = extract_subtrajectories(mem, n_subtrajectories, n_steps + n_warmup_steps, False)
        h = train_predictor(vae, all_predictor, n_actions, trajs, n_epochs, n_steps, n_warmup_steps)
        histories.append(h)

    for i, pred in enumerate(complete_predictors_list):
        pred.save_weights(f'predictors/predictor_{i}')
    with open('predictors/train_stats', 'wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_predictor_weights(complete_predictors_list, env_names, plots=False):
    graph_names = ['sp_env_0', 'sp_env_0+1', 'sp_env_0+1+2', 'ap_env_0', 'ap_env_0+1', 'ap_env_0+1+2']
    graph_colors = ['r', 'g', 'b', 'y', 'y', 'y']
    styles = ['-', '-', '-', '-.', '--', ':']

    for i, pred in enumerate(complete_predictors_list):
        pred.load_weights(f'predictors/predictor_{i}')

    if plots:
        with open('predictors/train_stats', 'rb') as handle:
            histories = pickle.load(handle)

        n_epochs, n_subtrajectories, n_steps, n_warmup_steps = histories.pop(0)

        fig = plt.figure(figsize=(14, 6))
        plt.suptitle(f'epochs: {n_epochs}, subtrajectories: {n_subtrajectories}, timesteps: {n_steps}, warmup: {n_warmup_steps}')
        plt.tight_layout()

        plt.subplot(131)
        plt.title(f'Prediction loss')
        for graph_name, col, style, hist in zip(graph_names, graph_colors, styles, histories):
            plt.plot(hist['loss'], color=col, linestyle=style, label=graph_name)

        plt.subplot(132)
        plt.title(f'Prediction mean absolute error')
        for graph_name, col, style, hist in zip(graph_names, graph_colors, styles, histories):
            plt.plot(hist['mean_absolute_error'], color=col, linestyle=style, label=graph_name)

        plt.legend()
        plt.show()


def test_predictor(predictor, mem, use_env_idx, vae, pos_finder, n_actions, n_steps, n_warmup_steps, video_title):
    predictor.predict_steps = n_steps

    trajectories = extract_subtrajectories(mem, 1, n_steps + n_warmup_steps - 1, False)

    obs = trajectories['s'] / 255
    next_obs = trajectories['s_'] / 255
    one_hot_acts = tf.one_hot(trajectories['a'], n_actions).numpy()
    end_pos = trajectories['pos']

    if use_env_idx:
        env_idx = trajectories['env'][0, 0, np.newaxis, np.newaxis]
    else:
        env_idx = np.zeros((1, 1), dtype=np.int32)

    encoded_obs = vae.encode(obs)
    encoded_next_obs = vae.encode(next_obs)
    encoded_start_obs = encoded_obs[:, 0:n_warmup_steps]
    targets = next_obs[0, n_warmup_steps - 1:]
    end_pos = end_pos[0, n_warmup_steps - 1:]

    #print([env_idx.shape, encoded_start_obs.shape, one_hot_acts.shape])

    # do rollout
    rollout = predictor.predict([env_idx, encoded_start_obs, one_hot_acts], n_steps)
    decoded_rollout_obs = np.clip(vae.decode(rollout), 0, 1)

    # remove batch dimension since it's 1 anyway
    next_obs = next_obs[0]
    decoded_rollout_obs = decoded_rollout_obs[0]

    # find player pos
    analyzed_player_pos = pos_finder.predict(decoded_rollout_obs).astype(int)
    pos_diff = np.abs(end_pos - analyzed_player_pos)
    pos_diff /= 2
    # debug_img = np.repeat(pos_diff, 3, axis=1).reshape(-1, 1, 2, 3)
    diff_img = np.clip(np.abs(decoded_rollout_obs - targets), 0, 1)

    trajectory_video([targets, decoded_rollout_obs, diff_img], ['true', 'predicted', 'difference'], overall_title=video_title)


def sfa_test():
    env = gym.make('Gridworld-room-v0')
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    latent_size = 16

    #memory = gen_data([env], episodes=100, file_name='sfa_test_train_data.npy')
    #simulate_agent_memories = gen_mixed_memories(memory, file_name='simulate_agent_memories.npy')
    memory = load_env_samples(['sfa_test_train_data.npy'])[0][0]
    print(memory.shape)

    vae = vae_net(obs_shape, latent_size)

    #load_vae_weights(vae, memory, plots=True)
    #train_vae(vae, memory, epochs=100)
    load_vae_weights(vae, memory, plots=False)
    #test_vae(vae, memory)

    traj = extract_subtrajectories(memory, 1, 800, warn=False, random=False)[0]
    reconstructed = np.clip(vae.predict(traj['s'] / 255), 0, 1)
    #anim = trajectory_video([traj['s'] / 255, reconstructed], ['true', 'reconstructed'])
    embedded_data = vae.encode(traj['s'] / 255)

    #i_f = 7
    #fake_data = np.repeat(embedded_data[0, np.newaxis], 300, axis=0)
    #fake_data[:, i_f] = np.linspace(-15, 15, 300)
    #reconstructed = np.clip(vae.decode(fake_data), 0, 1)
    #anim = trajectory_video([reconstructed], ['change feature'])
    #quit()

    #fig, axes = plt.subplots(16, 1, figsize=(16, 10))
    #for i, ax in enumerate(axes.flatten()):
    #    ax.plot(embedded_data[:, i], c='b')
    #    ax.plot(traj['pos'], c='r', linestyle='--')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    embedded_transformer = sksfa.SFA(n_components=3)
    pf = PolynomialFeatures(degree=3)
    extracted_features_embedded = embedded_transformer.fit_transform(pf.fit_transform(embedded_data))
    plt.plot(extracted_features_embedded)
    plt.show()

    plt.plot(traj['pos'])
    plt.show()
    plt.plot(embedded_data)
    plt.show()

    #transformer = sksfa.SFA(n_components=1)
    #data = traj['s'] / 255
    #extracted_features = transformer.fit_transform(data)
    #plt.plot(extracted_features)
    #plt.show()


def generalization_test():
    n_envs = 100
    envs = [gym.make('Gridworld-room-v0') for _ in range(n_envs)]
    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    latent_size = 16

    #memories = gen_data(envs, episodes=10, file_name='generalization_test_vae_train_data.npy')
    #simulate_agent_memories = gen_mixed_memories(memories, file_name='simulate_agent_memories.npy')
    memories = load_env_samples(['generalization_test_vae_train_data.npy'])[0]

    vae = vq_vae_net(obs_shape, latent_size)
    all_predictors = predictor_nets_2(latent_size, n_actions, 1)

    #load_vae_weights(vae, memories[-2:], plots=True)
    train_vae(vae, memories[:-1], epochs=50)
    load_vae_weights(vae, memories, plots=False)
    #test_vae(vae, memories)

    sfa_transformer = sksfa.SFA(n_components=3)
    traj = extract_subtrajectories(memories[-1], 1, 200, False)[0]
    reconstructed = np.clip(vae.predict(traj['s'] / 255), 0, 1)
    #reconstructed = np.clip(vae.decode(data[..., np.newaxis]), 0, 1)
    #extracted_features = sfa_transformer.fit_transform(data)

    #plt.plot(extracted_features)
    #plt.show()
    anim = trajectory_video([traj['s'] / 255, reconstructed], ['true', 'reconstructed'])

    #train_predictors_2(all_predictors, vae, memories, simulate_agent_memories, n_actions, n_epochs=80, n_subtrajectories=15000, n_steps=5, n_warmup_steps=1)





def split_predictor_test():
    #env_names = ['Gridworld-partial-room-v0', 'Gridworld-partial-room-v1', 'Gridworld-partial-room-v2']
    env_names = ['Gridworld-partial-random-v0']
    envs = [gym.make(env_name) for env_name in env_names]

    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    latent_size = 16

    pos_finder = pos_finder_net(obs_shape)
    #vae = vae_net(obs_shape, latent_size)
    vae = vq_vae_net(obs_shape, latent_size)
    all_predictors = predictor_nets_2(latent_size, n_actions, len(env_names))

    #memories = gen_data(envs, episodes=200, file_name='vae_train_data.npy')
    #simulate_agent_memories = gen_mixed_memories(memories, file_name='simulate_agent_memories.npy')
    memories, simulate_agent_memories = load_env_samples(['vae_train_data.npy', 'simulate_agent_memories.npy'])

    #train_pos_finder_net(pos_finder, memories, epochs=10)
    #load_pos_finder_weights(pos_finder, memories, pixel_size=2)

    train_vae(vae, memories, epochs=1)
    load_vae_weights(vae, memories, plots=True)

    #load_predictor_weights(all_predictors, env_names, plots=True)
    train_predictors_2(all_predictors, vae, memories, simulate_agent_memories, n_actions, n_epochs=80, n_subtrajectories=15000, n_steps=5, n_warmup_steps=1)
    load_predictor_weights(all_predictors, env_names, plots=True)

    #test_predictor(all_predictors[0], memories[0], True, vae, pos_finder, n_actions, n_steps=200, n_warmup_steps=1, video_title='Predictor 0')
    #test_predictor(all_predictors[0], memories[1], True, vae, pos_finder, n_actions, n_steps=200, n_warmup_steps=1, video_title='Predictor 1')
    #test_predictor(all_predictors[0], memories[2], True, vae, pos_finder, n_actions, n_steps=200, n_warmup_steps=1, video_title='Predictor 2')

    #test_predictor(all_predictors[1], memories[0], False, vae, pos_finder, n_actions, n_steps=200, n_warmup_steps=1, video_title='All-Predictor Memory 0')
    #test_predictor(all_predictors[1], memories[1], False, vae, pos_finder, n_actions, n_steps=200, n_warmup_steps=1, video_title='All-Predictor Memory 0 + 1')
    #test_predictor(all_predictors[1], memories[2], False, vae, pos_finder, n_actions, n_steps=200, n_warmup_steps=1, video_title='All-Predictor Memory 0 + 1 + 2')

    n_steps = 500
    n_trajectories = 200
    pred_0_perf = measure_performance_2(all_predictors[0], memories[0], True, vae, pos_finder, n_actions, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    pred_1_perf = measure_performance_2(all_predictors[0], memories[1], True, vae, pos_finder, n_actions, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    pred_2_perf = measure_performance_2(all_predictors[0], memories[2], True, vae, pos_finder, n_actions, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)

    all_pred_pref_0 = measure_performance_2(all_predictors[1], memories[0], False, vae, pos_finder, n_actions, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    all_pred_pref_1 = measure_performance_2(all_predictors[1], memories[1], False, vae, pos_finder, n_actions, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)
    all_pred_pref_2 = measure_performance_2(all_predictors[1], memories[2], False, vae, pos_finder, n_actions, n_steps=n_steps, n_warmup_steps=1, n_trajectories=n_trajectories)

    plt.plot(pred_0_perf, label='multihead_mem_0')
    plt.plot(pred_1_perf, label='multihead_mem_1')
    plt.plot(pred_2_perf, label='multihead_mem_2')
    plt.plot(all_pred_pref_0, label='all_predictor_mem_0', linestyle='--')
    plt.plot(all_pred_pref_1, label='all_predictor_mem_1', linestyle='--')
    plt.plot(all_pred_pref_2, label='all_predictor_mem_2', linestyle='--')
    plt.legend()
    plt.show()


def one_predictor_test():
    env = gym.make('Gridworld-room-v0')
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    latent_size = 16

    pos_finder = pos_finder_net(obs_shape)
    vae = vae_net(obs_shape, latent_size)

    memories, simulate_agent_memories = load_env_samples()
    load_pos_finder_weights(pos_finder, memories, pixel_size=2)
    load_vae_weights(vae, memories)

    intermediate_lws = [64, 64]
    lstm_lw = 256
    input_shape = (latent_size, n_actions)
    predictor = AutoregressivePredictor(input_shape, latent_size, intermediate_lws=intermediate_lws, lstm_lw=lstm_lw)
    predictor.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])#, run_eagerly=True)

    n_subtrajectories = 20000
    n_steps = 5
    n_epochs = 20
    n_warmup_steps = 1
    trajs = extract_subtrajectories(memories[0], n_subtrajectories, n_steps, False)
    h = train_predictor(vae, predictor, n_actions, trajs, n_epochs, n_steps, n_warmup_steps)
    predictor.save_weights('single_pred_weights')
    for k, v in h.items():
        plt.plot(v, label=k)
    plt.show()

    test_predictor(predictor, memories[0], vae, pos_finder, n_actions, n_steps=500, n_warmup_steps=1)


def measure_performance(predictor, mem, vae, pos_finder, n_actions, n_steps, n_warmup_steps, n_trajectories):
    predictor.predict_steps = n_steps

    trajectories = extract_subtrajectories(mem, n_trajectories, n_steps + n_warmup_steps - 1, False)

    obs = trajectories['s'] / 255
    next_obs = trajectories['s_'] / 255
    one_hot_acts = tf.one_hot(trajectories['a'], n_actions).numpy()
    end_pos = trajectories['pos']

    #encoded_obs = vae.encode(obs[:, :n_warmup_steps])
    #encoded_start_obs = encoded_obs[:, 0:n_warmup_steps]
    encoded_start_obs = vae.encode(obs[:, :n_warmup_steps])
    targets = next_obs[:, 0:n_steps]
    end_pos = end_pos[:, n_warmup_steps - 1:]

    # do rollout
    rollout = predictor.predict([encoded_start_obs, one_hot_acts], n_steps)
    decoded_rollout_obs = np.clip(vae.decode(rollout), 0, 1)

    # find player pos
    #analyzed_player_pos = pos_finder.predict(decoded_rollout_obs).astype(int)
    #pos_diff = np.abs(end_pos - analyzed_player_pos)
    #pos_diff /= 2
    # debug_img = np.repeat(pos_diff, 3, axis=1).reshape(-1, 1, 2, 3)
    diff_img = np.clip(np.abs(decoded_rollout_obs - targets), 0, 1)

    return diff_img.mean(axis=(0, 2, 3, 4))


def measure_performance_2(predictor, mem, use_env_idx, vae, pos_finder, n_actions, n_steps, n_warmup_steps, n_trajectories):
    predictor.predict_steps = n_steps

    trajectories = extract_subtrajectories(mem, n_trajectories, n_steps + n_warmup_steps - 1, False)

    obs = trajectories['s'] / 255
    next_obs = trajectories['s_'] / 255
    one_hot_acts = tf.one_hot(trajectories['a'], n_actions).numpy()
    end_pos = trajectories['pos']

    if use_env_idx:
        env_idx = trajectories['env'][:, 0]
    else:
        env_idx = np.zeros_like(trajectories['env'][:, 0])

    #encoded_obs = vae.encode(obs[:, :n_warmup_steps])
    #encoded_start_obs = encoded_obs[:, 0:n_warmup_steps]
    encoded_start_obs = vae.encode(obs[:, :n_warmup_steps])
    targets = next_obs[:, 0:n_steps]
    end_pos = end_pos[:, n_warmup_steps - 1:]

    # do rollout
    rollout = predictor.predict([env_idx, encoded_start_obs, one_hot_acts], n_steps)
    decoded_rollout_obs = np.clip(vae.decode(rollout), 0, 1)

    # find player pos
    #analyzed_player_pos = pos_finder.predict(decoded_rollout_obs).astype(int)
    #pos_diff = np.abs(end_pos - analyzed_player_pos)
    #pos_diff /= 2
    # debug_img = np.repeat(pos_diff, 3, axis=1).reshape(-1, 1, 2, 3)
    diff_img = np.clip(np.abs(decoded_rollout_obs - targets), 0, 1)

    return diff_img.mean(axis=(0, 2, 3, 4))


def vq_vae_net(obs_shape, latent_size):
    vae = VqVAE(obs_shape, latent_size, beta=1, kl_loss_factor_mul=1.000001)
    vae.compile(optimizer=tf.optimizers.Adam())
    return vae


if __name__ == '__main__':
    #run_basic_test()
    #one_predictor_test()
    generalization_test()
    #sfa_test()
    #split_predictor_test()



