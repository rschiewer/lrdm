from dataclasses import dataclass
from yamldataclassconfig import create_file_path_field
from yamldataclassconfig.config import YamlDataClassConfig
from pathlib import Path
import tensorflow as tf
import yaml
from typing import Union, List


class MultiYamlDataClassConfig(YamlDataClassConfig):

    def load(self, file_paths: Union[Path, str, List[Path], List[str]] = None, path_is_absolute: bool = False):
        """
        This method loads from YAML file to properties of self instance.
        Why doesn't load when __init__ is to make the following requirements compatible:
        1. Access config as global
        2. Independent on config for development or use config for unit testing when unit testing
        """
        if file_paths is None:
            file_paths = self.FILE_PATH

        if type(file_paths) is Path or type(file_paths) is str:
            file_paths = [file_paths]

        dict_config = {}
        for file_path in file_paths:
            dict_config.update(yaml.full_load(Path(file_path).read_text('UTF-8')))
        self.__dict__.update(self.__class__.schema().load(dict_config).__dict__)


@dataclass
class ExperimentConfig(MultiYamlDataClassConfig):
    tf_eager_mode: bool = None
    model_summaries: bool = None
    neptune_project_name: str = None

    env_setting: str = None
    env_n_samples_per_env: int = None
    env_mix_memory_fraction: Union[float, List[float]] = None
    env_sample_mem_path_stub: str = None
    env_mix_mem_path_stub: str = None
    env_name_concat: str = None

    vae_n_train_steps: int = None
    vae_n_steps_per_epoch: int = None
    vae_batch_size: int = None
    vae_commitment_cost: float = None
    vae_n_embeddings: int = None
    vae_d_embeddings: int = None
    vae_frame_stack: int = None
    vae_weights_path: str = None
    vae_train_stats_path: str = None

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
    pred_train_stats_path: str = None

    ctrl_n_runs: int = None
    ctrl_n_plan_steps: int = None
    ctrl_n_rollouts: int = None
    ctrl_n_iterations: int = None
    crtl_top_perc: float = None
    ctrl_gamma: float = None
    ctrl_do_mpc: bool = None
    ctrl_render: bool = None

    #FILE_PATH: Path = create_file_path_field(Path(__file__).parent / 'config_general.yml')


def load_config_files(file_paths):
    dict_config = {}
    for file_path in file_paths:
        dict_config.update(yaml.full_load(Path(file_path).read_text('UTF-8')))




# check if all directories for logging and storing stuff exist, generate if not present
#for k, v in vars(CONFIG).items():
#    if type(v) is str and '/' in v:
#        p = Path(v)
#        if not p.is_dir():
#            p.mkdir()


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
CONFIG.load(['config_general.yml', 'config_control.yml', 'config_predictor_1_mdl.yml', 'config_vae.yml'])
