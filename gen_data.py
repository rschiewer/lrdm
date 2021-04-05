from replay_memory_tools import gen_data, gen_mixed_memory
from config_loader import CONFIG
from train_tools import gen_environments

if __name__ == '__main__':
     env_names, envs, env_info = gen_environments(CONFIG.env_setting)
     sample_mem_paths = [CONFIG.env_sample_mem_path_stub + env_name for env_name in env_names]
     mix_mem_path = CONFIG.env_mix_mem_path_stub + CONFIG.env_name_concat.join(env_names)

     memories = gen_data(envs, env_info, samples_per_env=CONFIG.env_n_samples_per_env, file_paths=sample_mem_paths)
     gen_mixed_memory(memories, [1, 1, 1], file_path=mix_mem_path)
