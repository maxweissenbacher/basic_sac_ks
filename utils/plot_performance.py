import matplotlib.pyplot as plt
import numpy as np
import wandb


def load_runs_from_wandb_project_SAC(path):
    api = wandb.Api()
    run = api.run(path)
    if not run.state == "finished":
        raise ValueError(f"Run with ID {run.id} is not finished. Skipping this run.")
    nu = eval(run.config['env'])['nu']
    num_envs = eval(run.config['logger'])['num_eval_envs']
    all_rewards = []
    for env_id in range(num_envs):
        rewards = []
        for i, row in run.history(keys=[f"eval/reward_{env_id}"]).iterrows():
            rewards.append(row[f"eval/reward_{env_id}"])
        all_rewards.append(rewards)

    return nu, np.asarray(all_rewards)


def load_runs_from_wandb_project_DREAMER(path):
    print(f"WARNING: we normalise the Dreamer data by dividing by sqrt(64)=8. This should probably be fixed. BEWARE")
    norm = np.sqrt(64)
    api = wandb.Api()
    run = api.run(path)
    if not run.state == "finished":
        raise ValueError(f"Run with ID {run.id} is not finished. Skipping this run.")
    nu = run.config["KS"]["nu"]
    num_envs = run.config["envs"]["amount"]
    all_rewards = []
    for env_id in range(num_envs):
        rewards = []
        for i, row in run.history(keys=[f"rollout_eval_episode/rollout_reward_{env_id}"]).iterrows():
            rewards.append(row[f"rollout_eval_episode/rollout_reward_{env_id}"])
        all_rewards.append(rewards)

    return nu, np.asarray(all_rewards)/norm


def intersperse_lists(list1, list2):
    return [item for pair in zip(list1, list2) for item in pair]


def load_config_from_wandb_project(path):
    api = wandb.Api(timeout=30)
    for run in api.runs(path=path):
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        return run.config
    return None


if __name__ == '__main__':
    # Enable Latex for plot and choose font family
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    # ---------------------------
    # Data loading
    # ---------------------------

    # Load metrics from WandB
    team_name = "why_are_all_the_good_names_taken_aaa"
    project_name = "basic_sac_trained_on_L_12_eval_on_L_12"
    d_sac = {}
    d_sac['train L12 eval L60'] = load_runs_from_wandb_project_SAC(path='/'.join([team_name, project_name, '6ed1flhe']))
    d_sac['train L12 eval L31'] = load_runs_from_wandb_project_SAC(path='/'.join([team_name, project_name, '3khujipp']))
    d_sac['train L12 eval L22'] = load_runs_from_wandb_project_SAC(path='/'.join([team_name, project_name, 'w1wee5dz']))
    d_sac['train L12 eval L18'] = load_runs_from_wandb_project_SAC(path='/'.join([team_name, project_name, 'rb3tgm4q']))
    d_sac['train L12 eval L12'] = load_runs_from_wandb_project_SAC(path='/'.join([team_name, project_name, 'bta2goy5']))

    project_name = "Dreamer_KS_generalizability"
    d_dreamer = {}
    d_dreamer['train L12 eval L60'] = load_runs_from_wandb_project_DREAMER(path='/'.join([team_name, project_name, 'ugtni2p7']))
    d_dreamer['train L12 eval L31'] = load_runs_from_wandb_project_DREAMER(path='/'.join([team_name, project_name, '6jkfrbdm']))
    d_dreamer['train L12 eval L22'] = load_runs_from_wandb_project_DREAMER(path='/'.join([team_name, project_name, 'uoq3aiiz']))
    d_dreamer['train L12 eval L18'] = load_runs_from_wandb_project_DREAMER(path='/'.join([team_name, project_name, 'nytomo85']))
    d_dreamer['train L12 eval L12'] = load_runs_from_wandb_project_DREAMER(path='/'.join([team_name, project_name, 'ctmx5x3l']))

    # ---------------------------
    # Plotting
    # ---------------------------
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        'font.size': '15',
    })

    last_rewards_sac = {}
    last_rewards_dreamer = {}
    mean_rewards_sac = {}
    mean_rewards_dreamer = {}

    for key, (nu, rewards) in d_sac.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for env_id in range(rewards.shape[0]):
            plt.plot(rewards[env_id, :], label=f'env {env_id}')
        #plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Environment steps')
        plt.ylabel('Energy')
        plt.title(f'nu={nu}')
        plt.savefig(f'SAC {key}.png')
        plt.close()

        last_rewards_sac[str(int(2*np.pi/np.sqrt(nu)))] = rewards[:, -1]
        mean_rewards_sac[str(int(2 * np.pi / np.sqrt(nu)))] = np.mean(rewards, axis=-1)

    for key, (nu, rewards) in d_dreamer.items():
        # temporary fix for the 0 value
        rewards = rewards[:, 1:]
        # remove this once fixed
        fig, ax = plt.subplots(figsize=(10, 5))
        for env_id in range(rewards.shape[0]):
            plt.plot(rewards[env_id, :], label=f'env {env_id}')
        #plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Environment steps')
        plt.ylabel('Energy')
        plt.title(f'nu={nu}')
        plt.savefig(f'Dreamer {key}.png')
        plt.close()

        last_rewards_dreamer[str(int(2*np.pi/np.sqrt(nu)))] = rewards[:, -1]
        mean_rewards_dreamer[str(int(2 * np.pi / np.sqrt(nu)))] = np.mean(rewards, axis=-1)

    # Plot boxplots for last rewards
    fig, ax = plt.subplots(figsize=(10, 5))
    plt_list = []
    lbl_list = []
    for key in last_rewards_sac.keys():
        plt_list.append(np.abs(last_rewards_sac[key]))
        plt_list.append(np.abs(last_rewards_dreamer[key]))
        lbl_list.append(f"{key}")
        lbl_list.append(f"{key}")
    box = plt.boxplot(
        plt_list,
        labels=lbl_list,
        patch_artist=True,
    )
    # Color the boxes: red for sac, blue for dreamer
    colors = ['red', 'blue']
    for i, boxplot in enumerate(box['boxes']):
        boxplot.set_facecolor(colors[i % 2])
        boxplot.set_alpha(0.5)

    plt.yscale('log')
    plt.ylabel('Final reward')
    plt.title('Transfer learning (trained on L=12.5) \n Red = SAC, Blue = Dreamer')
    plt.xlabel('L (rounded to nearest integer)')
    plt.savefig('boxplot_last_reward_trained_on_L12.png')
    plt.show()
    plt.close()

    # Plot boxplots for mean rewards
    fig, ax = plt.subplots(figsize=(10, 5))
    plt_list = []
    lbl_list = []
    for key in mean_rewards_sac.keys():
        plt_list.append(np.abs(mean_rewards_sac[key]))
        plt_list.append(np.abs(mean_rewards_dreamer[key]))
        lbl_list.append(f"{key}")
        lbl_list.append(f"{key}")
    box = plt.boxplot(
        plt_list,
        labels=lbl_list,
        patch_artist=True,
    )
    # Color the boxes: red for sac, blue for dreamer
    colors = ['red', 'blue']
    for i, boxplot in enumerate(box['boxes']):
        boxplot.set_facecolor(colors[i % 2])
        boxplot.set_alpha(0.5)

    plt.yscale('log')
    plt.ylabel('Mean episodic reward')
    plt.title('Transfer learning (trained on L=12.5) \n Red = SAC, Blue = Dreamer')
    plt.xlabel('L (rounded to nearest integer)')
    plt.savefig('boxplot_mean_reward_trained_on_L12.png')
    plt.show()
    plt.close()



