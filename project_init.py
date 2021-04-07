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
CONFIG.load(['config_general.yml', 'config_control.yml', 'config_predictor_1_mdl.yml', 'config_vae.yml'])
