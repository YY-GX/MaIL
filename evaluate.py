import hydra
import logging
import os
import numpy as np
import torch
import imgaug.parameters as iap
from imgaug import augmenters as iaa
import argparse
from omegaconf import OmegaConf
import pickle
import wandb

current_working_directory = os.getcwd()
os.chdir(os.environ['PYTHONPATH'])
from libero.libero.envs import *
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark
os.chdir(current_working_directory)

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--model_folder_path", type=str,
                        default="/mnt/arc/yygx/pkgs_baselines/MaIL/checkpoints/separate_no_hand_ckpts/")
    parser.add_argument("--task_emb_dir", type=str,
                        default="/mnt/arc/yygx/pkgs_baselines/MaIL/task_embeddings/")

    args = parser.parse_args()
    return args


def eval(cfg, benchmark, task_embs, task_idx, agent, seed):
    # data augmentation
    aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(cfg.aug_factor), size_px=8),[255])

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite]()
    task_bddl_file = task_suite.get_task_bddl_file_path(task_idx)
    file_name = os.path.basename(task_bddl_file).split('.')[0]
    task_emb = task_embs[file_name]
    init_states = task_suite.get_task_init_states(task_idx)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }

    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(cfg.num_episode)]
    )

    agent.reset()
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state=init_states[0])

    num_success = 0
    dones = [False] * cfg.num_episode

    dummy = np.zeros((cfg.num_episode, 7))
    dummy[:, -1] = -1.0  # set the last action to -1 to open the gripper
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)


    with torch.no_grad():
        for j in range(cfg.max_step_per_episode):
            agentview_rgb = obs["agentview_image"]

            if cfg.data_aug:
                agentview_rgb = aug(image=agentview_rgb)

            all_actions = np.zeros(7)
            for each_agentview_rgb in agentview_rgb:
                state = (each_agentview_rgb, None, task_emb)
                action = agent.predict(state)[0]
                all_actions = np.vstack([all_actions, action])
            all_actions = all_actions[1:, :]
            obs, r, done, _ = env.step(all_actions)

            # check whether succeed
            for k in range(cfg.num_episode):
                dones[k] = dones[k] or done[k]
            if all(dones):
                break
    for k in range(cfg.num_episode):
        num_success += int(dones[k])

    env.close()

    return num_success / cfg.num_episode


def main() -> None:
    args = parse_args()

    def add_resolver(x, y):
        return x + y
    OmegaConf.register_new_resolver("add", add_resolver)
    cfg = OmegaConf.load(f"{args.model_folder_path}/multirun.yaml")

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="online",
        config=wandb.config
    )

    with open(f"{args.task_emb_dir}/{cfg.task_suite}.pkl", 'rb') as f:
        task_embs = pickle.load(f)

    benchmark = get_benchmark(cfg.task_suite)(cfg.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    tasks_succ_ls = []

    for task_idx in range(n_manip_tasks):
        task_name = benchmark.get_task_names()[task_idx]
        print(f">> Task Name: {task_name}")
        OmegaConf.resolve(cfg.agents)
        agent = hydra.utils.instantiate(cfg.agents, task_idx=task_idx)
        # Load checkpoints
        agent.load_pretrained_model(args.model_folder_path, f"last_ddpm_task_idx_{task_idx}.pth")
        # Eval pre-trained agent in Libero simu env
        sr = eval(cfg, benchmark, task_embs, task_idx, agent, seed=args.seed)
        print(f">> Success Rate for {task_name}: {sr}")
        tasks_succ_ls.append(sr)
        np.save(f"{args.model_folder_path}/succ_list_seed_{args.seed}.npy", np.array(tasks_succ_ls))

    print(f">> SR List: {tasks_succ_ls}")
    np.save(args.model_folder_path, np.array(tasks_succ_ls))


if __name__ == "__main__":
    main()
