from dataclasses import dataclass
from yamldataclassconfig import create_file_path_field
from yamldataclassconfig.config import YamlDataClassConfig
from pathlib import Path
import tensorflow as tf
from typing import Union, List


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


@dataclass
class ExperimentConfig(YamlDataClassConfig):
    tf_eager_mode: bool = None
    model_summaries: bool = None

    env_setting: str = None
    env_n_samples_per_env: int = None
    env_mix_memory_fraction: Union[float, List[float]] = None
    env_sample_mem_path_stub: str = None
    env_mix_mem_path_stub: str = None
    env_name_concat: str = None

    vae_n_train_steps: int = None
    vae_batch_size: int = None
    vae_commitment_cost: float = None
    vae_n_embeddings: int = None
    vae_d_embeddings: int = None
    vae_frame_stack: int = None
    vae_weights_path: str = None

    pred_n_train_steps: int = None
    pred_n_steps_per_epoch: int = None
    pred_n_trajectories: int = None
    pred_n_traj_steps: int = None
    pred_n_warmup_steps: int = None
    pred_pad_trajectories: bool = None
    pred_det_filters: int = None
    pred_prob_filters: int = None
    pred_decider_lw: int = None
    pred_n_models: int = None
    pred_batch_size: int = None
    pred_tb_log: bool = None
    pred_weights_path: str = None

    FILE_PATH: Path = create_file_path_field(Path(__file__).parent / 'config.yml')


CONFIG = ExperimentConfig()
CONFIG.load()


def gen_sample_mem_paths(env_names):
    return [CONFIG.env_sample_mem_path_stub + env_name for env_name in env_names]

def gen_mix_mem_path(env_names):
    return CONFIG.env_mix_mem_path_stub + CONFIG.env_name_concat.join(env_names)

def gen_vae_weights_path(env_names):
    return CONFIG.vae_weights_path + '_and_'.join(env_names)

def gen_predictor_weights_path(env_names):
    return CONFIG.pred_weights_path + '_and_'.join(env_names) + '_' + str(CONFIG.pred_n_models)

