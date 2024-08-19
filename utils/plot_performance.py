import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd


def load_runs_from_wandb_project(path, algorithm):
    api = wandb.Api()
    df = pd.DataFrame()
    for run in api.runs(path=path):
        rewards = []
        last_rewards = []
        cae_errors = []
        cae_absolute_errors = []
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        if not algorithm == run.name[:3]:
            continue
        num_sensors = eval(run.config['env'])['num_sensors']
        nu = eval(run.config['env'])['nu']
        num_actuators = eval(run.config['env'])['num_actuators']
        num_envs = eval(run.config['env'])['num_envs']
        use_cae = True if eval(run.config['env'])['path_to_cae_model'] else False
        for i, row in run.history(keys=["eval/reward"]).iterrows():
            rewards.append(row["eval/reward"])
        for i, row in run.history(keys=["eval/last_reward"]).iterrows():
            last_rewards.append(row["eval/last_reward"])
        if use_cae:
            for i, row in run.history(keys=["eval/cae_relative_L2_error"]).iterrows():
                cae_errors.append(row["eval/cae_relative_L2_error"])
            for i, row in run.history(keys=["eval/cae_absolute_L2_error"]).iterrows():
                cae_absolute_errors.append(row["eval/cae_absolute_L2_error"])
        else:
            cae_errors = len(rewards) * [0.]
            cae_absolute_errors = len(rewards) * [0.]
        df[run.name, use_cae, 'reward'] = rewards
        df[run.name, use_cae, 'errors'] = cae_errors
        df[run.name, use_cae, 'abs_errors'] = cae_absolute_errors
        #df[run.id, use_cae, 'last_reward'] = last_rewards

    return df


if __name__ == '__main__':
    # Enable Latex for plot and choose font family
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    no_memory_color = 'xkcd:cornflower blue'
    attention_memory_color = 'xkcd:coral'

    # ---------------------------
    # Data loading
    # ---------------------------

    # Load metrics for attention from WandB
    project_name = "LatentRL_solarsweep1"
    df_list = {}
    df_list['SAC'] = load_runs_from_wandb_project(
        path="why_are_all_the_good_names_taken_aaa/" + project_name,
        algorithm='SAC'
    )
    df_list['PPO'] = load_runs_from_wandb_project(
        path="why_are_all_the_good_names_taken_aaa/" + project_name,
        algorithm='PPO'
    )

    df_list_errors = {}
    df_list_absolute_errors = {}
    for key, df in df_list.items():
        df_cae = df[[c for c in df.columns if c[1] and c[2] == 'reward']]
        df_no_cae = df[[c for c in df.columns if not c[1] and c[2] == 'reward']]
        df_list[key] = {'CAE': df_cae, 'NO CAE': df_no_cae}
        df_list_errors[key] = df[[c for c in df.columns if c[1] and c[2] == 'errors']]
        df_list_absolute_errors[key] = df[[c for c in df.columns if c[1] and c[2] == 'abs_errors']]

    # Interpolate the errors to be the same length
    max_len = max([len(d) for d in df_list_errors.values()])
    for key, df in df_list_errors.items():
        if len(df) < max_len:
            x = np.linspace(0, 1, num=max_len)
            xp = np.linspace(0, 1, num=len(df))
            new_df = pd.DataFrame()
            for col in df.columns:
                new_df[col] = np.interp(x, xp, df[col])
            df_list_errors[key] = new_df

    # ---------------------------
    # Plotting
    # ---------------------------
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        'font.size': '20',
    })

    # Plot performance as a function of time
    algorithm = 'PPO'
    color1 = 'xkcd:cornflower blue'
    color2 = 'xkcd:coral'
    colors = {'CAE': color1, 'NO CAE': color2}
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, metric in df_list[algorithm].items():
        ax.plot(metric.abs().mean(axis=1), label=key, color=colors[key])

        ax.fill_between(
            range(len(metric.mean(axis=1))),
            metric.abs().mean(axis=1) - 1.96 * metric.abs().sem(axis=1),  # df_base_sens.abs().min(axis=1),
            metric.abs().mean(axis=1) + 1.96 * metric.abs().sem(axis=1),  # df_base_sens.abs().max(axis=1),
            color=colors[key], alpha=.2)

        #ax.plot(metric.abs(), color=colors[model], alpha=0.25)

    """
    ax.set_xticks(
        range(0, len(metric.min(axis=1)), 4),
        [f'{(x + 1) * 25}M' for x in range(0, len(metric.min(axis=1)), 4)]
    )
    """
    ax.set_xlabel('Solver steps')
    ax.set_ylabel("Energy " + r"$\displaystyle E$")
    # ax.set_ylabel('$\displaystyle L^2$ norm')
    ax.set_yscale('log')
    # plt.title(f"$\displaystyle L^2$ norm of KS solution per solver timestep")
    plt.legend()
    # plt.savefig('l2norm_filled.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()

    # Plot CAE errors as a function of time
    color1 = 'xkcd:cornflower blue'
    color2 = 'xkcd:coral'
    colors = {'SAC': color1, 'PPO': color2}
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, metric in df_list_errors.items():
        ax.plot(metric.abs().mean(axis=1), label=key, color=colors[key])

        ax.fill_between(
            range(len(metric.mean(axis=1))),
            metric.abs().mean(axis=1) - 1.96 * metric.abs().sem(axis=1),  # df_base_sens.abs().min(axis=1),
            metric.abs().mean(axis=1) + 1.96 * metric.abs().sem(axis=1),  # df_base_sens.abs().max(axis=1),
            color=colors[key], alpha=.2)

        # ax.plot(metric.abs(), color=colors[model], alpha=0.25)

    """
    ax.set_xticks(
        range(0, len(metric.min(axis=1)), 4),
        [f'{(x + 1) * 25}M' for x in range(0, len(metric.min(axis=1)), 4)]
    )
    """
    ax.set_xlabel('Solver steps')
    ax.set_ylabel("Relative " + r"$\displaystyle L^2$" + " error")
    # ax.set_ylabel('$\displaystyle L^2$ norm')
    # plt.title(f"$\displaystyle L^2$ norm of KS solution per solver timestep")
    plt.legend()
    # plt.savefig('l2norm_filled.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()

    # Plot CAE absolute errors as a function of time
    color1 = 'xkcd:cornflower blue'
    color2 = 'xkcd:coral'
    colors = {'SAC': color1, 'PPO': color2}
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, metric in df_list_absolute_errors.items():
        ax.plot(metric.abs().mean(axis=1), label=key, color=colors[key])

        ax.fill_between(
            range(len(metric.mean(axis=1))),
            metric.abs().mean(axis=1) - 1.96 * metric.abs().sem(axis=1),  # df_base_sens.abs().min(axis=1),
            metric.abs().mean(axis=1) + 1.96 * metric.abs().sem(axis=1),  # df_base_sens.abs().max(axis=1),
            color=colors[key], alpha=.2)

        # ax.plot(metric.abs(), color=colors[model], alpha=0.25)

    """
    ax.set_xticks(
        range(0, len(metric.min(axis=1)), 4),
        [f'{(x + 1) * 25}M' for x in range(0, len(metric.min(axis=1)), 4)]
    )
    """
    ax.set_xlabel('Solver steps')
    ax.set_ylabel("Absolute " + r"$\displaystyle L^2$" + " error")
    # ax.set_ylabel('$\displaystyle L^2$ norm')
    # plt.title(f"$\displaystyle L^2$ norm of KS solution per solver timestep")
    plt.legend()
    # plt.savefig('l2norm_filled.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()
