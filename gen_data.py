from replay_memory_tools import gen_data, gen_mixed_memory
from project_init import CONFIG
from train_tools import gen_environments

if __name__ == '__main__':
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
