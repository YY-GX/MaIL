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
import multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
# os.chdir(current_working_directory)

log = logging.getLogger(__name__)

is_use_hand = True
is_multitask = True

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


def main() -> None:
    args = parse_args()

    def add_resolver(x, y):
        return x + y
    def now(format: str):
        return datetime.now().strftime(format)
    OmegaConf.register_new_resolver("add", add_resolver)
    OmegaConf.register_new_resolver("now", now)
    cfg = OmegaConf.load(f"{args.model_folder_path}/multirun.yaml")

    # cpu things
    # HydraConfig.instance().set_config(cfg)
    # job_num = HydraConfig.get().job.num
    # job_num = hydra.core.hydra_config.HydraConfig.get().job.num
    # num_cpu = mp.cpu_count()
    # cpu_set = list(range(num_cpu))
    # current_num = int(job_num % 4)
    # assign_cpus = cpu_set[current_num * cfg.n_cores:current_num * cfg.n_cores + cfg.n_cores]

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="online",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(args.model_folder_path, f"last_ddpm.pth")
    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent, cpu_set=None, epoch=888,
                       is_save=True, folder=args.model_folder_path, task_suite=args.task_suite, seed=args.seed)


if __name__ == "__main__":
    main()
