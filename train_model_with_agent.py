from project_init import CONFIG, gen_mix_mem_path, gen_vae_weights_path, gen_predictor_weights_path
from tools import gen_environments, vq_vae_net, predictor_net
from simulated_env import *

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network, encoding_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import TimeStep
import neptune.new as neptune
import tensorflow as tf


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def gen_dqn_agent(collect_env: gym.Env, learn_env: gym.Env, vae: VectorQuantizerEMAKeras):
    replay_buffer_capacity = 100000  # @param {type:"integer"}

    fc_layer_params = [64, 64]
    conv_layer_params = [(32, 3, 2)]

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    gamma = 0.99

    num_atoms = 51  # @param {type:"integer"}
    min_q_value = -3  # @param {type:"integer"}
    max_q_value = 3 # @param {type:"integer"}
    collect_traj_len = 10  # @param {type:"integer"}
    agent_n_step_update = 5

    learn_py_env = suite_gym.wrap_env(learn_env)
    learn_env = tf_py_environment.TFPyEnvironment(learn_py_env)

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        learn_env.observation_spec(),
        learn_env.action_spec(),
        num_atoms=num_atoms,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)#,
        #preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        learn_env.time_step_spec(),
        learn_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        epsilon_greedy=None,
        boltzmann_temperature=0.5,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=agent_n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=train_step_counter)
    agent.initialize()


    # VAE dataset with real environment images
    space_o = collect_env.observation_space['o']
    obs_spec = BoundedArraySpec(space_o.shape, space_o.dtype, minimum=space_o.low, maximum=space_o.high)
    obs_spec._shape = tf.TensorShape(space_o.shape)  # get around bug in BoundedArraySpec

    space_task = collect_env.observation_space['task']
    task_spec = BoundedArraySpec(space_task.shape, space_task.dtype, minimum=space_task.low, maximum=space_task.high)
    task_spec._shape = tf.TensorShape(space_task.shape)  # get around bug in BoundedArraySpec

    vae_collect_data_spec = trajectory.Trajectory(
        agent.collect_data_spec.step_type,
        {'o': obs_spec, 'task': task_spec},
        agent.collect_data_spec.action,
        agent.collect_data_spec.policy_info,
        agent.collect_data_spec.next_step_type,
        agent.collect_data_spec.reward,
        agent.collect_data_spec.discount
    )

    vae_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=vae_collect_data_spec,
        batch_size=learn_env.batch_size,
        max_length=replay_buffer_capacity)

    @tf.function
    def vae_collect_step_fn(environment, policy):
        time_step = environment.current_time_step()
        # embed observation to make it compatible with the agent
        embedded_obs = vae.encode_to_vectors(cast_and_normalize_images(time_step.observation['o']))
        transformed_time_step = TimeStep(
            time_step.step_type,
            time_step.reward,
            time_step.discount,
            embedded_obs
        )
        # generate behavior with transformed observation
        action_step = policy.action(transformed_time_step)
        next_time_step = environment.step(action_step.action)

        # store original trajectory in vae replay buffer
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        vae_replay_buffer.add_batch(traj)

    vae_dataset = vae_replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size,
                                               num_steps=collect_traj_len + 1).prefetch(3)

    # agent replay buffer for learning in the simulated environment
    agent_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=learn_env.batch_size,
        max_length=replay_buffer_capacity)

    @tf.function
    def agent_collect_step_fn(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        agent_replay_buffer.add_batch(traj)

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    agent_dataset = agent_replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size,
                                                   num_steps=agent_n_step_update + 1).prefetch(3)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    return agent, vae_collect_step_fn, vae_dataset, agent_collect_step_fn, agent_dataset, agent_replay_buffer


def train_vae(vae, dataset: tf.data.Dataset, epochs=1, steps_per_epoch=200):
    vae_callback = tf.keras.callbacks.EarlyStopping(monitor='total_loss', min_delta=1e-6, patience=3)
    train_dataset = dataset.map(lambda traj, info: cast_and_normalize_images(traj.observation['o']))
    vae.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[vae_callback])


def prepare_pred_data(traj, info, vae, warmup_steps, training=None):
    obs = traj.observation['o'][:, :warmup_steps]
    actions = tf.cast(traj.action[:, :-1, tf.newaxis], tf.int32)
    next_obs = traj.observation['o'][:, 1:]
    rewards = tf.cast(traj.reward[:, :-1, tf.newaxis], tf.float32)
    done = tf.cast(traj.is_last()[:, :-1, tf.newaxis], tf.float32)
    task_id = tf.cast(traj.observation['task'][:, :-1], tf.float32)

    encoded_obs = tf.cast(vae.encode_to_indices(cast_and_normalize_images(obs)), tf.float32)
    encoded_next_obs = tf.cast(vae.encode_to_indices(cast_and_normalize_images(next_obs)), tf.float32)
    if training:
        return (encoded_obs, actions), (encoded_next_obs, rewards, done, task_id)
    else:
        return encoded_obs, actions


def train_predictor(pred: RecurrentPredictor, vae: VectorQuantizerEMAKeras, dataset: tf.data.Dataset, warmup_steps=1,
                    epochs=1, steps_per_epoch=200):
    pred_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=3)
    train_dataset = dataset.map(lambda traj, info: prepare_pred_data(traj, info, vae, warmup_steps, training=True))
    pred.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[pred_callback])


"""
def train_rl_algo(agent, pred: RecurrentPredictor, vae: VectorQuantizerEMAKeras, dataset: tf.data.Dataset, warmup_steps):
    train_dataset = dataset.map(lambda traj, info: prepare_pred_data(traj, info, vae, warmup_steps, training=False))
    simulation_input = next(iter(train_dataset))
    simulated_batch = pred.predict(simulation_input)
    embedded_obs = vae.indices_to_embeddings(tf.concat([simulation_input[0], simulated_batch[0][:, :-1]], axis=1))
    train_trajs = []
    for (_, r, done, _) in simulated_batch:
        #t = trajectory.Trajectory(observation=embedded_obs, )
        pass
    train_loss = agent.train(train_trajs)

    return train_loss
"""

#@tf.function
def train_agent_in_simulation(agent, learn_env, agent_dataset, agent_collect_step_fn, n_samples):
    for _ in range(n_samples):
        agent_collect_step_fn(learn_env, agent.collect_policy)
    experience, unused_info = next(iter(agent_dataset))
    train_loss = agent.train(experience)
    return train_loss


def train_agent(agent, pred, vae, collect_env_gym, learn_env_gym, eval_env_gym, vae_dataset,
                vae_collect_step_fn, agent_dataset, agent_collect_step_fn, agent_replay_buffer, initial_collect_steps,
                collect_steps_per_iteration, num_iterations, num_eval_episodes, log_interval, eval_interval,
                train_vae_interval, train_pred_interval, run):
    collect_py_env = suite_gym.wrap_env(collect_env_gym)
    collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
    learn_py_env = suite_gym.wrap_env(learn_env_gym)
    learn_env = tf_py_environment.TFPyEnvironment(learn_py_env)
    eval_py_env = suite_gym.wrap_env(eval_env_gym)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # random baseline
    random_policy = random_tf_policy.RandomTFPolicy(eval_env.time_step_spec(), eval_env.action_spec())
    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # initial data collection
    for _ in range(initial_collect_steps):
        vae_collect_step_fn(collect_env, random_policy)

    # train vae and predcitor initially
    vae.load_weights('pretrained_vae_weights')
    pred.load_weights('pretrained_pred_weights')
    #train_vae(vae, vae_dataset, epochs=50)
    #train_predictor(pred, vae, vae_dataset, epochs=80)
    #vae.save_weights('pretrained_vae_weights')
    #pred.save_weights('pretrained_pred_weights')

    # start training
    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            vae_collect_step_fn(collect_env, agent.collect_policy)

        # TODO: exchange this for generating new data in simulated env
        # Sample a batch of data from the buffer and update the agent's network.
        #train_loss = train_rl_algo(agent, pred, vae, vae_dataset, 1)
        agent_replay_buffer.clear()
        train_loss = train_agent_in_simulation(agent, learn_env, agent_dataset, agent_collect_step_fn, 512)
        step = agent.train_step_counter.numpy()

        # train vae and predcitor a bit
        if step % train_vae_interval == 0:
            train_vae(vae, vae_dataset, epochs=15)
        if step % train_pred_interval == 0:
            train_predictor(pred, vae, vae_dataset, epochs=15)

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            if run:
                run['average_return'].log(avg_return)
                run['step'].log(step)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)

    return returns


if __name__ == '__main__':
    env_names, learn_envs, env_info = gen_environments(CONFIG.env_setting)
    _, collect_envs, _ = gen_environments(CONFIG.env_setting)
    _, eval_envs, _ = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    predictor_weights_path = gen_predictor_weights_path(env_names)

    rand_seed = 42
    num_iterations = 50000  # @param {type:"integer"}
    initial_collect_steps = 5000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    log_interval = 100  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 200  # @param {type:"integer"}
    train_vae_interval = 500
    train_pred_interval = 500

    if CONFIG.env_setting == 'gridworld_3_rooms_rand_starts':
        print('Setting 100 steps time limit for all envs')
        for env in learn_envs:
            env.time_limit = 100
        for env in collect_envs:
            env.time_limit = 100
        for env in eval_envs:
            env.time_limit = 100
        print('Deactivating random starts for eval env')
        for env in eval_envs:
            env.player_random_start = False

    # instantiate vae and load trained weights
    vae = vq_vae_net(obs_shape=env_info['obs_shape'],
                     n_embeddings=CONFIG.vae_n_embeddings,
                     d_embeddings=CONFIG.vae_d_embeddings,
                     train_data_var=1,
                     commitment_cost=CONFIG.vae_commitment_cost,
                     frame_stack=CONFIG.vae_frame_stack,
                     summary=CONFIG.model_summaries,
                     tf_eager_mode=CONFIG.tf_eager_mode)
    #load_vae_weights(vae=vae, weights_path=vae_weights_path)

    # instantiate predictor
    pred = predictor_net(n_actions=env_info['n_actions'],
                         obs_shape=env_info['obs_shape'],
                         n_envs=len(learn_envs),
                         vae=vae,
                         det_filters=CONFIG.pred_det_filters,
                         prob_filters=CONFIG.pred_prob_filters,
                         decider_lw=CONFIG.pred_decider_lw,
                         n_models=CONFIG.pred_n_models,
                         tensorboard_log=CONFIG.pred_tb_log,
                         summary=CONFIG.model_summaries,
                         tf_eager_mode=CONFIG.tf_eager_mode)
    #pred.load_weights(predictor_weights_path)

    if CONFIG.neptune_project_name:
        run = neptune.init(project=CONFIG.neptune_project_name)
        run['parameters'] = {k: v for k, v in vars(CONFIG).items()}
        run['sys/tags'].add('agent_training')
        if not CONFIG.tf_eager_mode:
            run['predictor_params'] = pred.count_params()
            run['vae_params'] = vae.count_params()
    else:
        run = None

    tf.random.set_seed(rand_seed)
    np.random.seed(rand_seed)

    agent_learn_env = MultiSimulatedLatentSpaceEnv(learn_envs, pred, vae, None, 0.9)
    agent_eval_env = MultiLatentSpaceEnv(eval_envs, vae, None)
    agent_collect_env = MultiEnv(collect_envs, [0, 1, 2])

    agent, vae_collect_step_fn, vae_dataset, agent_collect_step_fn, agent_dataset, agent_replay_buffer =\
        gen_dqn_agent(agent_collect_env, agent_learn_env, vae)

    returns = train_agent(agent, pred, vae, agent_collect_env, agent_learn_env, agent_eval_env, vae_dataset, vae_collect_step_fn,
                agent_dataset, agent_collect_step_fn, agent_replay_buffer,
                initial_collect_steps, collect_steps_per_iteration, num_iterations, num_eval_episodes, log_interval,
                eval_interval, train_vae_interval, train_pred_interval, run)



    #for i_task, (collect_env, learn_env, eval_env) in enumerate(zip(collect_envs[0:1], learn_envs[0:1], eval_envs[0:1])):
    #    learn_env = SimulatedLatentSpaceEnv(learn_env, pred, vae, None)
    #    eval_env = LatentSpaceEnv(eval_env, vae, None)
    #    agent, collect_step_fn, vae_dataset = gen_dqn_agent(collect_env, learn_env, vae)
    #    train_agent(agent, pred, vae, collect_env, learn_env, eval_env, vae_dataset, collect_step_fn,
    #                initial_collect_steps, collect_steps_per_iteration, num_iterations, num_eval_episodes, log_interval,
    #                eval_interval, run)
