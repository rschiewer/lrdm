import matplotlib.pyplot as plt
from matplotlib import animation, rc
import time
from predictors import *
from vae import *
from blockworld import *
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import gym
import warnings
import os
from see_rnn import *

#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
#os.environ['LD_LIBRARY_PATH']='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\CUPTI\lib64'
#os.environ['PATH']='C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin'
#os.environ['PATH']+='C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'


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



def gen_sin_time_series(n_time_series, n_steps, dimension=1, min_val=0, max_val=10, max_phase_shift=5):
    train_data = []
    for i_ts in range(n_time_series):
        phase_shift = np.random.random() * max_phase_shift
        x = np.arange(min_val + phase_shift, max_val + phase_shift, (max_val - min_val) / n_steps)
        train_data.append(np.sin(x)[:, np.newaxis])
    return np.array(train_data)


def gen_lstm_network(in_shape):
    x_in = layers.Input(shape=in_shape)
    x = layers.Dense(32, activation='relu')(x_in)
    x = layers.LSTM(32, return_sequences=False, return_state=True)(x)
    x = layers.Dense(32, activation='relu')(x)

    mdl = keras.Model(inputs=x_in, outputs=x)
    mdl.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    return mdl


def run_basic_test():

    n_time_series = 5000
    epochs = 200
    data_timesteps = 70
    train_type_0 = False
    train_type_1 = False
    train_type_2 = True

    #tf.autograph.set_verbosity(2, alsologtostdout=True)
    #data = gen_time_series(n_time_series=n_time_series, n_steps=timesteps, lower=0, upper=40)
    data = gen_sin_time_series(n_time_series=n_time_series, n_steps=data_timesteps, dimension=3)
    pred = AutoregressiveProbabilisticPredictor(input_sizes=[1, 1], output_sizes=1, lstm_lw=64, intermediate_lws=[64])
    pred.compile(loss='mse', optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])#, run_eagerly=True)
    #pred.summary()
    #print(pred.evaluate_step.pretty_printed_concrete_signatures())

    pred.predict_steps = 30
    warmup_steps = 2

    if train_type_0:
        # first training; use real input data in every step, predictor should learn what to remember from state to state here
        x0 = data[:, 0:pred.predict_steps]
        x1 = np.ones(shape=(n_time_series, pred.predict_steps, 1))
        y = data[:, 1:pred.predict_steps + 1]

        history = pred.fit([x0, x1], y, epochs=15, batch_size=64, validation_split=0.15, shuffle=True)
        #for k, v in history.history.items():
        #    plt.plot(v, label=k)
        #plt.legend()
        #plt.show()

    if train_type_1:
        # second training; produce static rollout and use for training
        x0 = data[:, 0:pred.predict_steps]
        x1 = np.ones(shape=(n_time_series, pred.predict_steps, 1))
        x0 = pred.predict([x0, x1])  # generate rollouts and use them as input during training
        y = data[:, 1:pred.predict_steps + 1]

        history = pred.fit([x0, x1], y, epochs=15, batch_size=64, validation_split=0.15, shuffle=True)
        #for k, v in history.history.items():
        #    plt.plot(v, label=k)
        #plt.legend()
        #plt.show()

    if train_type_2:
        # third training; only provide starting states and use full autoregressive rollouts during fit
        x0 = data[:, 0:warmup_steps]
        x1 = np.ones(shape=(n_time_series, pred.predict_steps, 1))
        y = data[:, 1:pred.predict_steps + 1]

        history = pred.fit([x0, x1], y, epochs=epochs, batch_size=64, validation_split=0.15, shuffle=True)
        #for k, v in history.history.items():
        #    plt.plot(v, label=k)
        #plt.legend()
        #plt.show()

    # test -------------------------------------------------------------------------------------------------------------

    x0 = data[0:3, 0:warmup_steps]
    x1 = np.ones(shape=(3, pred.predict_steps, 1))
    y = data[0:3, 1: pred.predict_steps + 1]

    predictions = pred.predict([x0, x1])

    distances = np.abs(predictions[warmup_steps:] - y[warmup_steps:]).mean(axis=0).squeeze()
    plt.plot(distances)
    plt.show()

    warmup_x_vals = list(range(warmup_steps))
    x_vals = list(range(warmup_steps, pred.predict_steps))
    plt.scatter(x_vals, predictions[0][warmup_steps:], label='predictions', marker='+')
    plt.scatter(x_vals, y[0][warmup_steps:], label='real data', marker='x')
    plt.scatter(warmup_x_vals, x0[0], label='warmup data', marker='x')
    plt.legend()
    plt.show()

    plt.scatter(x_vals, predictions[1][warmup_steps:], label='predictions', marker='+')
    plt.scatter(x_vals, y[1][warmup_steps:], label='real data', marker='x')
    plt.scatter(warmup_x_vals, x0[1], label='warmup data', marker='x')
    plt.legend()
    plt.show()

    plt.scatter(x_vals, predictions[2][warmup_steps:], label='predictions', marker='+')
    plt.scatter(x_vals, y[2][warmup_steps:], label='real data', marker='x')
    plt.scatter(warmup_x_vals, x0[2], label='warmup data', marker='x')
    plt.legend()
    plt.show()


def cluster_latent_space(vae, memory):
    encoded_obs = vae.encode(memory['s'] / 255)
    n_clusters = list(range(2, 50))
    clusterings, db_scores = [], []

    for n_c in n_clusters:
        kmeans = KMeans(n_clusters=n_c).fit(encoded_obs)
        db_score = davies_bouldin_score(encoded_obs, kmeans.labels_)
        clusterings.append(kmeans)
        db_scores.append(db_score)

    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax0.scatter(n_clusters, [kmeans.score(encoded_obs) for kmeans in clusterings], label='clustering error')
    #ax0.xticks(n_clusters, n_clusters)
    ax0.grid(True)
    ax0.legend()
    ax1.scatter(n_clusters, db_scores, label='db score', c='r')
    #ax1.xticks(n_clusters, n_clusters)
    ax1.grid(True)
    ax1.legend()
    plt.show()


def plot_latent_space(vae, memory):
    from sklearn.manifold import TSNE
    encoded_obs = vae.encode(memory['s'] / 255)
    envs = memory['env']
    embedded = TSNE(n_components=2).fit_transform(encoded_obs)

    labels = []
    for i_e in envs:
        if i_e == 0:
            labels.append('r')
        elif i_e == 1:
            labels.append('g')
        elif i_e == 2:
            labels.append('b')
        else:
            raise RuntimeError('Unknown environment')

    plt.scatter(embedded[:, 0], embedded[:, 1], c=labels)
    plt.show()
