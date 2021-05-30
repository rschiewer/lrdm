import neptune.new as neptune
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(palette='muted')


def plot_avg_steps(project_name, env_names, filter_ids, filter_tags, alternative_task_names=None, alternative_labels=None):
    project = neptune.get_project(project_name)
    runs = project.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    # collect data
    df = pd.DataFrame(columns=['run_id', 'n_models', 'environment', 'steps_mean', 'steps_std', 'r_mean', 'r_std'])
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])

        for env_name in env_names:
            steps = run[f'{env_name}/steps'].fetch_values()['value'].to_numpy()
            rewards = run[f'{env_name}/rewards'].fetch_values()['value'].to_numpy()
            row = {'run_id': run_metadata['sys/id'], 'n_models': run["parameters/pred_n_models"].fetch(),
                   'environment': env_name, 'steps_mean': steps.mean(), 'steps_std': steps.std(),
                   'r_mean': rewards.mean(), 'r_std': rewards.std()}
            df = df.append(row, ignore_index=True)

    # plotting
    x = np.arange(len(env_names))
    n_ids = len(filter_ids)
    width = 0.35
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))


    n_exp_setups = len(df.groupby('n_models'))
    for _, grouped_by_env in df.groupby('environment'):
        for i, (name, grouped_by_n_models) in enumerate(grouped_by_env.groupby('n_models')):
            steps_mean = grouped_by_n_models['steps_mean'].mean()
            steps_std = grouped_by_n_models[]
            ax0.bar(x - width/2 * (n_exp_setups-1) + width * i, run_data['steps_mean'], width, yerr=run_data['steps_std'], label=label)
            ax1.bar(x - width/2 * (n_exp_setups-1) + width * i, run_data['r_mean'], width, yerr=run_data['r_std'], label=label)

    for i, id in enumerate(filter_ids):
        run_data = df.loc[df['run_id'] == id]
        if alternative_labels:
            label = alternative_labels[i]
        else:
            label = f'{run_data["n_models"].mean()}'
        ax0.bar(x - width/2 * (n_ids-1) + width * i, run_data['steps_mean'], width, yerr=run_data['steps_std'], label=label)
        ax1.bar(x - width/2 * (n_ids-1) + width * i, run_data['r_mean'], width, yerr=run_data['r_std'], label=label)
        ax0.grid(True, fillstyle='bottom', alpha=0.5)
        ax1.grid(True, fillstyle='bottom', alpha=0.5)

    ax0.set_ylabel('Average Steps')
    ax0.set_xticks(x)
    if alternative_task_names:
        ax0.set_xticklabels(alternative_task_names)
    else:
        ax0.set_xticklabels(env_names)

    ax1.set_ylabel('Average Rewards')
    ax1.set_xticks(x)
    if alternative_task_names:
        ax1.set_xticklabels(alternative_task_names)
    else:
        ax1.set_xticklabels(env_names)

    ax1.legend()
    fig.tight_layout()
    plt.show()


def plot_loss(project_name, filter_ids, filter_tags, log_scale=False, alternative_labels=None):
    project = neptune.get_project(project_name)
    runs = project.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    df = pd.DataFrame(columns=['run_id', 'n_models', 'loss'])
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])
        loss = run['metrics/epoch/loss'].fetch_values()['value'].to_numpy()
        row = {'run_id': run_metadata['sys/id'], 'n_models': run["parameters/pred_n_models"].fetch(), 'loss': loss}
        df = df.append(row, ignore_index=True)

    for i, row in enumerate(df.iterrows()):
        if alternative_labels:
            label = alternative_labels[i]
        else:
            label = f'{row[1]["n_models"]} models'
        plt.plot(row[1]['loss'], label=label)
    if log_scale:
        plt.yscale('log')
        plt.title('RDM loss (log scale)')
    else:
        plt.title('RDM loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


def plot_picker_loss(project_name, filter_ids, filter_tags, log_scale=False, alternative_labels=None):
    project = neptune.get_project(project_name)
    runs = project.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    df = pd.DataFrame(columns=['run_id', 'n_models', 'picker_loss'])
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])
        loss = run['metrics/epoch/picker_loss'].fetch_values()['value'].to_numpy()
        row = {'run_id': run_metadata['sys/id'], 'n_models': run["parameters/pred_n_models"].fetch(), 'picker_loss': loss}
        df = df.append(row, ignore_index=True)

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


if __name__ == '__main__':
    project_name = 'rschiewer/predictor'
    env_names = ['Gridworld-partial-room-v3', 'Gridworld-partial-room-v4', 'Gridworld-partial-room-v5']

    plot_avg_steps(project_name, env_names, ['PRED-643', 'PRED-642', 'PRED-666', 'PRED-667'], ['control'], ['Task 1', 'Task 2', 'Task 3'], ['monolithic RDM', 'multi RDM'])
    #plot_loss(project_name, ['PRED-635', 'PRED-637'], ['predictor'], False, ['monolithic RDM', 'multi RDM'])
    #plot_picker_loss(project_name, ['PRED-635', 'PRED-637'], ['predictor'], False, ['monolithic RDM', 'multi RDM'])
