from dataclasses import dataclass
from yamldataclassconfig import create_file_path_field
from yamldataclassconfig.config import YamlDataClassConfig
from pathlib import Path


@dataclass
class ExperimentConfig(YamlDataClassConfig):
    tf_eager_mode: bool = None
    model_summaries: bool = None

    env_setting: str = None
    env_n_samples_per_env: int = None
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


