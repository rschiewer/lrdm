from project_init import *
from tools import *
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
from gym.wrappers import TransformObservation
from tf_tools import InflateLayer2


def train_dqn_agent(simulated_env: gym.Env, real_env: gym.Env, seed: float):
    num_iterations = 50000  # @param {type:"integer"}

    initial_collect_steps = 10000  # @param {type:"integer"}
    transitions_allowed = 100000
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 100000  # @param {type:"integer"}

    fc_layer_params = (100,)
    conv_layer_params = [(32, 3, 2)]

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    gamma = 0.99
    log_interval = 200  # @param {type:"integer"}

    num_atoms = 51  # @param {type:"integer"}
    min_q_value = -20  # @param {type:"integer"}
    max_q_value = 20  # @param {type:"integer"}
    n_step_update = 5  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    # reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_py_env = suite_gym.wrap_env(simulated_env)
    eval_py_env = suite_gym.wrap_env(real_env)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    obs_shape = train_env.observation_spec()['o'].shape

    def combiner_fn(input):
        obs = tf.cast(input['o'], tf.float32)
        task_id = tf.cast(tf.expand_dims(input['task'], 1), tf.float32)
        tid_inflated = InflateLayer2(obs_shape[:2], 2)(task_id)
        x = tf.keras.layers.Concatenate()([obs, tid_inflated])
        return x

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        conv_layer_params=conv_layer_params,
        preprocessing_combiner=tf.keras.layers.Lambda(lambda input: combiner_fn(input)),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)
    transitions_collected = tf.compat.v2.Variable(0)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        epsilon_greedy=None,
        boltzmann_temperature=0.5,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        target_update_period= 1000,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=train_step_counter)
    agent.initialize()

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

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    def collect_step(environment, policy):
        transitions_collected.assign(transitions_collected.value() + 1)

        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    for _ in range(initial_collect_steps):
        collect_step(train_env, random_policy)

    # This loop is so common in RL, that we provide standard implementations of
    # these. For more details see the drivers module.

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            if transitions_collected.value() < transitions_allowed:
                collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: collected = {1}: loss = {2}'.format(step, transitions_collected.value(), train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)


if __name__ == '__main__':
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    _, eval_envs, _ = gen_environments(CONFIG.env_setting)
    mix_mem_path = gen_mix_mem_path(env_names)
    vae_weights_path = gen_vae_weights_path(env_names)
    predictor_weights_path = gen_predictor_weights_path(env_names)
    rand_seed = 42

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

    # train in simulated environment
    #env = MultiSimulatedLatentSpaceEnv(envs, pred, vae, [0, 1, 2], 0.9)
    #eval_env = MultiLatentSpaceEnv(envs, vae, [0, 1, 2])
    #train_dqn_agent(env, eval_env, rand_seed)

    # train on latent space env
    #env = MultiLatentSpaceEnv(envs, vae, [0, 1, 2])
    #eval_env = MultiLatentSpaceEnv(envs, vae, [0, 1, 2])
    #train_dqn_agent(env, eval_env, rand_seed)

    # train on original env
    env = CharToFloatObs(MultiEnv(envs, [0, 1, 2]))
    eval_env = CharToFloatObs(MultiEnv(envs, [0, 1, 2]))
    train_dqn_agent(env, eval_env, rand_seed)

    #for env, eval_env in zip(envs[0:1], eval_envs[0:1]):
        # train on simulated environment
        #env = SimulatedLatentSpaceEnv(env, pred, vae)
        #eval_env = LatentSpaceEnv(eval_env, vae)
        #train_dqn_agent(env, eval_env, rand_seed)

        # train on latent space env
        #env = LatentSpaceEnv(env, vae)
        #eval_env = LatentSpaceEnv(eval_env, vae)
        #train_dqn_agent(env, eval_env, rand_seed)

        # train on original env
        #env = CharToFloatObs(env)
        #eval_env = CharToFloatObs(eval_env)
        #train_dqn_agent(env, eval_env, rand_seed)

