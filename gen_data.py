from replay_memory_tools import gen_data, gen_mixed_memory
from project_init import CONFIG
from tools import gen_environments
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import ResizeObservation

if __name__ == '__main__':

    """env = ResizeObservation(gym.make('VizdoomTakeCover-v0'), (64, 64))
    print(env.observation_space)
    print(env.action_space.n)
    state_0 = np.copy(env.reset())
    for i in range(20):
        state_1, reward, done, info = env.step(2)
    diff_img = np.abs(state_0 - state_1)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(state_0)
    ax1.imshow(state_1)
    ax2.imshow(diff_img)
    plt.show()
    quit()"""
    env_names, envs, env_info = gen_environments(CONFIG.env_setting)
    sample_mem_paths = [CONFIG.env_sample_mem_path_stub + env_name for env_name in env_names]
    mix_mem_path = CONFIG.env_mix_mem_path_stub + CONFIG.env_name_concat.join(env_names)

    if type(CONFIG.env_mix_memory_fraction) is float or type(CONFIG.env_mix_memory_fraction) is int:
        mix_fractions = [CONFIG.env_mix_memory_fraction for _ in env_names]
    else:
        assert len(CONFIG.env_mix_memory_fraction) == len(env_names), 'Need either a single or one float per env'
        mix_fractions = CONFIG.env_mix_memory_fraction

    memories = gen_data(envs, env_info, samples_per_env=CONFIG.env_n_samples_per_env, file_paths=sample_mem_paths)
    gen_mixed_memory(memories, mix_fractions, file_path=mix_mem_path)
