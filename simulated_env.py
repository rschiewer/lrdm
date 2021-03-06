import numpy as np
import gym
import tensorflow as tf
from gym import spaces
from predictors import RecurrentPredictor
from keras_vq_vae import VectorQuantizerEMAKeras
from tools import cast_and_normalize_images, cast_and_unnormalize_images
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional


class TaskEnv(gym.Env):

    def __init__(self, orig_env: gym.Env, task_id: Union[int, None] = None):
        super(TaskEnv, self).__init__()
        if task_id is not None:
            assert task_id < 255, 'Maximum of 255 tasks'

        self._orig_env = orig_env
        self._task_id = task_id
        self.action_space = orig_env.action_space
        if task_id is None:
            self.observation_space = orig_env.observation_space
        else:
            self.observation_space = spaces.Dict({'o': orig_env.observation_space,
                                                  'task': spaces.Box(low=0, high=254, shape=(), dtype=np.uint8)})

    @property
    def unwrapped(self):
        return self._orig_env

    def _build_complete_obs(self, obs):
        if self._task_id is None:
            return tf.cast(obs, tf.float32)
        else:
            return {'o': tf.cast(obs, tf.float32), 'task': self._task_id}

    def step(self, action):
        o, r, done, info = self._orig_env.step(action)
        return self._build_complete_obs(o), r, done, info

    def reset(self):
        o = self._orig_env.reset()
        return self._build_complete_obs(o)

    def render(self, mode='human'):
        self._orig_env.render(mode)


class LatentSpaceEnv(TaskEnv):

    def __init__(self, orig_env: gym.Env, vae: VectorQuantizerEMAKeras, task_id: Union[int, None] = None):
        super(LatentSpaceEnv, self).__init__(orig_env, task_id)

        self._vae = vae
        self._plot_win = None
        self._current_state = None

        self.action_space = orig_env.action_space
        obs_shape = vae.compute_latent_shape(orig_env.observation_space.shape) + (vae.embedding_dim,)
        if task_id is None:
            self.observation_space = spaces.Box(low=0, high=vae.num_embeddings, shape=obs_shape, dtype=np.float32)
        else:
            self.observation_space = spaces.Dict(
                {'o': spaces.Box(low=0, high=vae.num_embeddings, shape=obs_shape, dtype=np.float32),
                 'task': spaces.Box(low=0, high=254, shape=(), dtype=np.uint8)})

    def _encode_image_obs(self, obs):
        normalized_obs = cast_and_normalize_images(obs)
        encoded_obs = self._vae.encode_to_vectors(normalized_obs[np.newaxis, ...])[0]
        return encoded_obs

    def _decode_image_obs(self, embedding):
        decoded = self._vae.decode_from_vectors(embedding[np.newaxis, np.newaxis, ..., 0])
        return cast_and_unnormalize_images(decoded[0, 0, ...])

    def reset(self):
        original_obs = self._orig_env.reset()
        embedded_obs = self._encode_image_obs(original_obs)
        self._current_state = embedded_obs
        return self._build_complete_obs(embedded_obs)

    def step(self, action):
        assert self.action_space.contains(action), f'Action {action} is not part of the action space'

        o, r, done, info = self._orig_env.step(action)
        o_embedded = self._encode_image_obs(o)
        self._current_state = o_embedded
        #reconstructed = self._decode_embedding(o_embedded)
        #plt.imshow(reconstructed)
        #plt.show()

        return self._build_complete_obs(o_embedded), r, done, info

    def render(self, mode='human'):
        state = self._current_state
        decoded_rollout_obs = cast_and_unnormalize_images(self._vae.decode_from_vectors(state[np.newaxis, ...]))[0, ...]
        if self._plot_win is None:
            self._plot_win = plt.imshow(decoded_rollout_obs.numpy())
            plt.pause(0.1)
            plt.ion()
            plt.show()
        else:
            self._plot_win.set_data(decoded_rollout_obs.numpy())
            plt.pause(0.1)
            plt.draw()


class SimulatedLatentSpaceEnv(LatentSpaceEnv):

    def __init__(self, orig_env: gym.Env, simulator: RecurrentPredictor, vae: VectorQuantizerEMAKeras,
                 task_id: int = None, done_threshold: float = 0.9):
        super(SimulatedLatentSpaceEnv, self).__init__(orig_env, vae, task_id)

        self._simulator = simulator
        self._done_threshold = done_threshold

    def step(self, action):
        assert self.action_space.contains(action), f'Action {action} is not part of the action space'

        in_o = tf.convert_to_tensor(self._current_state[np.newaxis, np.newaxis, ...])
        in_o = self._vae.embeddings_to_indices(in_o)
        in_a = tf.fill((1, 1, 1), action)
        o_predicted_indices, r_predicted, done_predicted, w_predictors = self._simulator.predict([in_o, in_a])
        o_predicted_embeddings = self._vae.indices_to_embeddings(o_predicted_indices)
        self._current_state = np.squeeze(o_predicted_embeddings)

        # remove batch and time dimensions of 1
        r_ret = np.squeeze(r_predicted)
        done_ret = np.squeeze(done_predicted)
        w_ret = np.squeeze(w_predictors)

        return self._build_complete_obs(self._current_state), r_ret, done_ret > self._done_threshold, {'predictor_weights:': w_ret}

    def render(self, mode='human'):
        state = self._current_state
        decoded_rollout_obs = cast_and_unnormalize_images(self._vae.decode_from_vectors(state[np.newaxis, ...]))[0, ...]
        if self._plot_win is None:
            self._plot_win = plt.imshow(decoded_rollout_obs.numpy())
            plt.pause(0.1)
            plt.ion()
            plt.show()
        else:
            self._plot_win.set_data(decoded_rollout_obs.numpy())
            plt.pause(0.1)
            plt.draw()


class MultiEnv(gym.Env):

    def __init__(self, orig_envs: List[gym.Env], task_ids: List[int] = None):
        super(MultiEnv, self).__init__()

        if task_ids is None:
            task_ids = [None] * len(orig_envs)
        else:
            assert len(orig_envs) == len(task_ids), 'If task_ids is not None, must have same length as number of envs'

        self._orig_envs = orig_envs
        self._task_ids = task_ids

        self._envs = self._gen_envs()
        self._active_env = np.random.default_rng().integers(0, len(self._envs))

        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space

        for env in self._envs:
            assert env.action_space == self.action_space, 'All envs must have the same action space'
            assert env.observation_space == self.observation_space, 'All envs must have the same observation space'

    def _gen_envs(self):
        return [TaskEnv(env, tid) for env, tid in zip(self._orig_envs, self._task_ids)]

    def reset(self, task_id: int = None):
        if task_id is None:
            self._active_env = np.random.default_rng().integers(0, len(self._envs))
        else:
            self._active_env = task_id
        return self._envs[self._active_env].reset()

    def step(self, action):
        return self._envs[self._active_env].step(action)

    def render(self, mode='human'):
        self._envs[self._active_env].render(mode)


class MultiLatentSpaceEnv(MultiEnv):

    def __init__(self, orig_envs: List[gym.Env], vae: VectorQuantizerEMAKeras, task_ids: List[int] = None):
        self._vae = vae
        super(MultiLatentSpaceEnv, self).__init__(orig_envs, task_ids)

    def _gen_envs(self):
        return [LatentSpaceEnv(env, self._vae, tid) for env, tid in zip(self._orig_envs, self._task_ids)]


class MultiSimulatedLatentSpaceEnv(MultiLatentSpaceEnv):

    def __init__(self, orig_envs: List[gym.Env], simulator: RecurrentPredictor, vae: VectorQuantizerEMAKeras, task_ids: List[int] = None,
                 done_threshold: float = 0.9):
        self._done_threshold = done_threshold
        self._simulator = simulator
        super(MultiSimulatedLatentSpaceEnv, self).__init__(orig_envs, vae, task_ids)

    def _gen_envs(self):
        return [SimulatedLatentSpaceEnv(env, self._simulator, self._vae, tid) for env, tid in zip(self._orig_envs, self._task_ids)]
