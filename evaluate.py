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
import datetime
import json

# current_working_directory = os.getcwd()
# os.chdir(os.environ['PYTHONPATH'])
from libero.libero.envs import *
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark, get_benchmark_dict, task_orders, find_keys_by_value
# os.chdir(current_working_directory)

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--is_osm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--task_order_index", type=int, default=0)
    parser.add_argument("--task_suite", type=str, default="libero_90")
    parser.add_argument("--model_folder_path", type=str,
                        default="/mnt/arc/yygx/pkgs_baselines/MaIL/checkpoints/separate_no_hand_ckpts/")
    parser.add_argument("--task_emb_dir", type=str,
                        default="/mnt/arc/yygx/pkgs_baselines/MaIL/task_embeddings/")

    args = parser.parse_args()
    return args


def create_index_mapping(dict_map):
    output_map = {}
    key_index = 0
    value_index = 0

    for key, values in dict_map.items():
        for _ in values:
            output_map[value_index] = key_index
            value_index += 1
        key_index += 1

    return output_map


def eval(cfg, task_embs, task_idx, agent, seed, is_osm, mapping, task_suite):
    # data augmentation
    aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(cfg.aug_factor), size_px=8),[255])

    task_suite = get_benchmark_dict()[task_suite]()
    task_bddl_file = task_suite.get_task_bddl_file_path(task_idx)
    file_name = os.path.basename(task_bddl_file).split('.')[0]
    if is_osm:
        task_ori = find_keys_by_value(mapping, file_name + ".bddl")[0]
        task_emb = task_embs[task_ori]
    else:
        task_emb = task_embs[file_name]
    init_states = task_suite.get_task_init_states(task_idx)
    indices = np.arange(cfg.simulation.num_episode) % init_states.shape[0]
    init_states_ = init_states[indices]
    print(f">> task_bddl_file: {task_bddl_file}")
    print("==========================================================")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }

    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(cfg.simulation.num_episode)]
    )

    agent.reset()
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state=init_states_)

    num_success = 0
    dones = [False] * cfg.simulation.num_episode

    dummy = np.zeros((cfg.simulation.num_episode, 7))
    dummy[:, -1] = -1.0  # set the last action to -1 to open the gripper
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)


    with torch.no_grad():
        for j in range(cfg.simulation.max_step_per_episode):
            agentview_rgb = [each_obs["agentview_image"] for each_obs in obs]

            if cfg.data_aug:
                agentview_rgb = [aug(image=rgb) for rgb in agentview_rgb]

            all_actions = np.zeros(7)
            for each_agentview_rgb in agentview_rgb:
                state = (each_agentview_rgb, None, task_emb)
                action = agent.predict(state)[0]
                all_actions = np.vstack([all_actions, action])
            all_actions = all_actions[1:, :]
            obs, r, done, _ = env.step(all_actions)

            # check whether succeed
            for k in range(cfg.simulation.num_episode):
                dones[k] = dones[k] or done[k]
            if all(dones):
                break
    for k in range(cfg.simulation.num_episode):
        num_success += int(dones[k])

    env.close()

    return num_success / cfg.simulation.num_episode


def main() -> None:
    args = parse_args()

    def add_resolver(x, y):
        return x + y
    def now(format: str):
        return datetime.now().strftime(format)
    OmegaConf.register_new_resolver("add", add_resolver)
    OmegaConf.register_new_resolver("now", now)
    cfg = OmegaConf.load(f"{args.model_folder_path}/multirun.yaml")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="online",
        config=wandb.config
    )

    with open(f"{args.task_emb_dir}/libero_90.pkl", 'rb') as f:
        task_embs = pickle.load(f)


    if args.is_osm:
        mapping_dir = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/mappings"
        mapping_pth = f"{mapping_dir}/{args.task_suite}.json"
        with open(mapping_pth, 'r') as json_file:
            mapping = json.load(json_file)
        index_mapping = create_index_mapping(mapping)


    benchmark = get_benchmark(args.task_suite)(args.task_order_index)
    task_id_ls = task_orders[args.task_order_index]

    tasks_succ_ls = []

    """
    task_id_ls is the list of evaluated task ids (original task may only have 44, 
    but modified tasks could be much more, e.g., 100, then len(task_id_ls) == 100)
    """
    for task_idx, task_id in enumerate(task_id_ls):
        print("===================== Start Evaluation =====================")
        if args.is_osm:
            model_index = index_mapping[task_id]  # model_index is the id for original model index
            print(f">> Load model checkpoint id: {model_index}")
            # TODO - Sarturday work on this

        task_name = benchmark.get_task_names()[task_id]
        print(f">> Task Name: {task_name}")
        OmegaConf.resolve(cfg.agents)
        agent = hydra.utils.instantiate(cfg.agents, task_idx=task_id)
        # Load checkpoints
        if args.is_osm:
            agent.load_pretrained_model(args.model_folder_path, f"last_ddpm_task_idx_{model_index}.pth")
        else:
            agent.load_pretrained_model(args.model_folder_path, f"last_ddpm_task_idx_{task_id}.pth")
            mapping = None
        # Eval pre-trained agent in Libero simu env
        sr = eval(cfg, task_embs, task_id, agent, seed=args.seed, is_osm=args.is_osm, mapping=mapping, task_suite=args.task_suite)
        print(f">> Success Rate for {task_name}: {sr}")
        tasks_succ_ls.append(sr)
        np.save(f"{args.model_folder_path}/succ_list_bm_{args.task_suite}_seed_{args.seed}.npy", np.array(tasks_succ_ls))

    print(f">> SR List: {tasks_succ_ls}")
    np.save(f"{args.model_folder_path}/succ_list_bm_{args.task_suite}_seed_{args.seed}.npy", np.array(tasks_succ_ls))


if __name__ == "__main__":
    main()
