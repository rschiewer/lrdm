from dataclasses import dataclass
from yamldataclassconfig import create_file_path_field
from yamldataclassconfig.config import YamlDataClassConfig
from pathlib import Path
import tensorflow as tf
from tools import ExperimentConfig

# check if all directories for logging and storing stuff exist, generate if not present
#for k, v in vars(CONFIG).items():
#    if type(v) is str and '/' in v:
#        p = Path(v)
#        if not p.is_dir():
#            p.mkdir()


# Global options and settings ------------------------------------------------------------------------------------------


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


CONFIG = ExperimentConfig()
CONFIG.load(['config_general.yml', 'config_control.yml', 'config_predictor_3_mdl.yml', 'config_vae.yml'])


def gen_sample_mem_paths(env_names):
    return [CONFIG.env_sample_mem_path_stub + env_name for env_name in env_names]


def gen_mix_mem_path(env_names):
    return CONFIG.env_mix_mem_path_stub + CONFIG.env_name_concat.join(env_names)


def gen_vae_weights_path(env_names):
    return CONFIG.vae_weights_path + '_and_'.join(env_names)


def gen_vae_train_stats_path(env_names):
    return CONFIG.vae_train_stats_path + '_and_'.join(env_names)


def gen_predictor_weights_path(env_names):
    return CONFIG.pred_weights_path + '_and_'.join(env_names) + '_' + str(CONFIG.pred_n_models)


def gen_predictor_train_stats_path(env_names):
    return CONFIG.pred_train_stats_path + '_and_'.join(env_names) + '_' + str(CONFIG.pred_n_models)