import neptune.new as neptune
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(palette='muted')


def plot_avg_steps(project_name, env_names, filter_ids, filter_tags, model_labels, task_names):
    project = neptune.get_project(project_name)
    runs = project.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    # collect data
    df = pd.DataFrame(columns=['run_id', 'n_models', 'environment', 'steps', 'reward'])
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])

        for env_name in env_names:
            steps = run[f'{env_name}/steps'].fetch_values()['value'].to_numpy()
            rewards = run[f'{env_name}/rewards'].fetch_values()['value'].to_numpy()
            for step, reward in zip(steps, rewards):
                row = {'run_id': run_metadata['sys/id'], 'n_models': run["parameters/pred_n_models"].fetch(),
                       'environment': env_name, 'steps': step, 'reward': reward}
                df = df.append(row, ignore_index=True)

    # plotting
    x = np.arange(len(env_names))
    width = 0.35
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    n_exp_setups = len(df.groupby('n_models'))
    for i, (name, grouped_by_n_models) in enumerate(df.groupby('n_models')):
        r_means = []
        r_stds = []
        steps_means = []
        steps_stds = []
        for _, grouped_by_env in grouped_by_n_models.groupby('environment'):
            steps_means.append(grouped_by_env['steps'].mean())
            steps_stds.append(grouped_by_env['steps'].std())
            r_means.append(grouped_by_env['reward'].mean())
            r_stds.append(grouped_by_env['reward'].std())
        ax0.bar(x - width/2 * (n_exp_setups-1) + width * i, steps_means, width, yerr=steps_stds, label=model_labels[name])
        ax1.bar(x - width/2 * (n_exp_setups-1) + width * i, r_means, width, yerr=r_stds, label=model_labels[name])
    ax0.grid(True, fillstyle='bottom', alpha=0.5)
    ax1.grid(True, fillstyle='bottom', alpha=0.5)

    ax0.set_title('Average Steps')
    ax0.set_ylabel('Average Steps (Lower is Better)')
    ax0.set_xticks(x)
    ax0.set_xticklabels(task_names)

    ax1.set_title('Average Rewards')
    ax1.set_ylabel('Average Rewards (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names)

    ax1.legend()
    fig.tight_layout()
    plt.savefig('rewards and steps per env.png', dpi=300)
    #plt.show()


def plot_loss(project_name, filter_ids, filter_tags, labels, log_scale=False):
    project = neptune.get_project(project_name)
    runs = project.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    df = pd.DataFrame(columns=['run_id', 'n_models', 'loss'])
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])
        loss = run['metrics/epoch/loss'].fetch_values()['value'].to_numpy()
        row = {'run_id': run_metadata['sys/id'], 'n_models': run["parameters/pred_n_models"].fetch(), 'loss': loss}
        df = df.append(row, ignore_index=True)

    fig = plt.figure(figsize=(6, 4))
    for i, (name, grouped_by_n_models) in enumerate(df.groupby('n_models')):
        loss_vals = grouped_by_n_models['loss'].to_numpy()
        loss_vals = np.array([arr for arr in loss_vals])
        x = range(len(loss_vals[0]))
        y = loss_vals.mean(axis=0)
        y_std = loss_vals.std(axis=0)
        plt.plot(x, y, label=labels[name])
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.3)
    if log_scale:
        plt.yscale('log')
        #plt.title('RDM loss (log scale)')
    else:
        pass
        #plt.title('RDM loss')
    plt.suptitle('RDM Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('RDM loss.png', dpi=300)
    #plt.show()


def plot_picker_loss(project_name, filter_ids, filter_tags, labels, log_scale=False):
    project = neptune.get_project(project_name)
    runs = project.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    df = pd.DataFrame(columns=['run_id', 'n_models', 'picker_loss'])
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])
        loss = run['metrics/epoch/picker_loss'].fetch_values()['value'].to_numpy()
        row = {'run_id': run_metadata['sys/id'], 'n_models': run["parameters/pred_n_models"].fetch(), 'picker_loss': loss}
        df = df.append(row, ignore_index=True)

    fig = plt.figure(figsize=(6, 4))
    for i, (name, grouped_by_n_models) in enumerate(df.groupby('n_models')):
        loss_vals = grouped_by_n_models['picker_loss'].to_numpy()
        loss_vals = np.array([arr for arr in loss_vals])
        x = range(len(loss_vals[0]))
        y = loss_vals.mean(axis=0)
        y_std = loss_vals.std(axis=0)
        plt.plot(x, y, label=labels[name])
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.3)
    if log_scale:
        plt.yscale('log')
        #plt.title('Task Classification Loss (log scale)')
    else:
        pass
        #plt.title('Task Classification Loss')
    plt.suptitle('Task Classification Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('TCN loss.png', dpi=300)
    #plt.show()

    """
    #fig, ax = plt.subplots()
    for i, row in enumerate(df.iterrows()):
        if alternative_labels:
            label = alternative_labels[i]
        else:
            label = f'{row[1]["n_models"]} models'
        plt.plot(row[1]['picker_loss'], label=label)
    if log_scale:
        plt.yscale('log')
        plt.title('TCN loss (log scale)')
    else:
        plt.title('TCN loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    """


if __name__ == '__main__':
    project_name = 'rschiewer/predictor'
    env_names = ['Gridworld-partial-room-v3', 'Gridworld-partial-room-v4', 'Gridworld-partial-room-v5']

    plot_avg_steps(project_name, env_names, ['PRED-643', 'PRED-642', 'PRED-666', 'PRED-667', 'PRED-680', 'PRED-679'],
                   ['control'], {1: 'monolithic RDM', 3: 'multi RDM'}, ['Task 1', 'Task 2', 'Task 3'])

    #plot_loss(project_name, ['PRED-635', 'PRED-637', 'PRED-660', 'PRED-659', 'PRED-676', 'PRED-677'], ['predictor'], {1: 'monolithic RDM', 3: 'multi RDM'}, False)
    #plot_picker_loss(project_name, ['PRED-635', 'PRED-637', 'PRED-660', 'PRED-659', 'PRED-676', 'PRED-677'], ['predictor'], {1: 'monolithic RDM', 3: 'multi RDM'}, False)
