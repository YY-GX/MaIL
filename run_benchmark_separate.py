import os
import logging
import random

import hydra
import numpy as np
import multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from agents.utils import sim_framework_path

# current_working_directory = os.getcwd()
# os.chdir(os.environ['PYTHONPATH'])
from libero.libero.envs import *
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark
# os.chdir(current_working_directory)

log = logging.getLogger(__name__)

print(torch.cuda.is_available())

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="config", config_name="benchmark_libero.yaml")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="online",
        config=wandb.config
    )


    # env_sim = hydra.utils.instantiate(cfg.simulation)

    job_num = hydra.core.hydra_config.HydraConfig.get().job.num

    num_cpu = mp.cpu_count()
    cpu_set = list(range(num_cpu))
    current_num = int(job_num % 4)
    assign_cpus = cpu_set[current_num * cfg.n_cores:current_num * cfg.n_cores + cfg.n_cores]

    benchmark = get_benchmark(cfg.task_suite)(cfg.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    for task_idx in range(n_manip_tasks):
        task_name = benchmark.get_task_names()[task_idx]
        print("===================== Start Training =====================")
        print(f"Start training task: {task_name}")
        print("==========================================================")
        agent = hydra.utils.instantiate(cfg.agents, task_idx=task_idx)
        for _ in tqdm(range(agent.epoch)):
            agent.train_single_vision_agent()

        agent.store_model_weights(store_path=f"/mnt/arc/yygx/pkgs_baselines/MaIL/checkpoints/{cfg.folder_name}", sv_name=f"last_ddpm_task_idx_{task_idx}.pth")
        print("===================== SAVING CHECKPOINTS =====================")
        print(f"Model saved for task {task_name} "
              f"at /mnt/arc/yygx/pkgs_baselines/MaIL/last_ddpm_task_idx_{task_idx}")
        print("==============================================================")

    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    main()
