import neptune.new as neptune
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    project_name = 'rschiewer/predictor'
    filter_tags = ['control']
    filter_ids = ['PRED-437', 'PRED-438']

    hd_proj = neptune.get_project(project_name)
    runs = hd_proj.fetch_runs_table(tag=filter_tags, id=filter_ids).to_pandas()
    print(f'Found {len(runs)} runs...')

    per_env = [[], [], []]
    for i, run_metadata in runs.iterrows():
        run = neptune.init(project=project_name, run=run_metadata['sys/id'])
        print(run_metadata['sys/id'])
        r0 = run['Gridworld-partial-room-v0/rewards'].fetch_values()
        r1 = run['Gridworld-partial-room-v1/rewards'].fetch_values()
        r2 = run['Gridworld-partial-room-v2/rewards'].fetch_values()

        per_env[0].append(r0['value'].to_numpy())
        per_env[1].append(r1['value'].to_numpy())
        per_env[2].append(r2['value'].to_numpy())

    per_env = np.array(per_env)

    split_pred_r_mean = per_env[:, 0].mean(axis=1)
    split_pred_r_std = per_env[:, 0].std(axis=1)
    mono_pred_r_mean = per_env[:, 1].mean(axis=1)
    mono_pred_r_std = per_env[:, 1].std(axis=1)

    labels = ['env 0', 'env 1', 'env 2']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, split_pred_r_mean, width, yerr=split_pred_r_std, label='mono')
    rects2 = ax.bar(x + width/2, mono_pred_r_mean, width, yerr=mono_pred_r_std, label='split')

    ax.set_ylabel('Average Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()




