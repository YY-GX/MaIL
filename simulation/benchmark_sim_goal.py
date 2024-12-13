import logging
import os
import cv2
import random
import numpy as np
import torch
import wandb
import robosuite
import multiprocessing as mp
from .base_sim import BaseSim


# current_working_directory = os.getcwd()
# os.chdir(os.environ['PYTHONPATH'])
from libero.libero.envs import *
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark, get_benchmark_dict, task_orders, find_keys_by_value
# os.chdir(current_working_directory)


import imgaug.parameters as iap
from imgaug import augmenters as iaa
import json

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.


# aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(0.02), size_px=8),
#                                         [255])



class MultiTaskSim(BaseSim):
    def __init__(self,
                 num_episode,
                 max_step_per_episode,
                 task_suite: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render,
                 n_cores,
                 camera_shape: tuple,
                 data_aug: bool = False,
                 aug_factor: float = 0.02,
                 task_id: int = 0,
                 nms: float = 0.1):
        super().__init__(seed, device, render, n_cores)

        # data augmentation
        self.data_aug = data_aug
        self.aug_factor = aug_factor
        self.aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(self.aug_factor), size_px=8),
                                                     [255])

        # according to the task_id, load the corresponding bddl file
        self.task_suite = task_suite

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render
        self.task_id = task_id

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0

    def eval_agent(self, agent, contexts, context_ind, success, pid, cpu_set, is_osm=False, task_suite_="single_step", is_use_hand=True):
        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        # env_ids = []

        print(f"contexts: {contexts}")

        for i, context in enumerate(contexts):

            # if context not in env_ids:
            # env_ids.append(context)

            task_suite = benchmark.get_benchmark_dict()[task_suite_]()

            task_bddl_file = task_suite.get_task_bddl_file_path(context)

            file_name = os.path.basename(task_bddl_file).split('.')[0]

            if is_osm:
                # TODO
                mapping_dir = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/mappings"
                mapping_pth = f"{mapping_dir}/{task_suite_}.json"
                with open(mapping_pth, 'r') as json_file:
                    mapping = json.load(json_file)
                task_ori = find_keys_by_value(mapping, file_name + ".bddl")[0]
                task_emb = self.task_embs[task_ori]
            else:
                task_emb = self.task_embs[file_name]

            # goal_images = self.goal_dicts[file_name]
            # goal_image = random.choice(goal_images)

            init_states = task_suite.get_task_init_states(context)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }

            env = OffScreenRenderEnv(**env_args)

            agent.reset()
            env.seed(self.seed)
            env.reset()
            # obs = env.set_init_state(init_state=init_states[context_ind[i]])
            # YY: I add this
            obs = env.set_init_state(init_state=init_states[0])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = obs["agentview_image"]

                # save_path = os.path.join("/home/i53/student/wang/OCIL/OCIL", f"{self.task_suite}", "images")
                # img = env.sim.render(camera_name="frontview", width=1280, height=800)[..., ::-1]
                # img = np.flip(img, axis=0)
                # cv2.imwrite(os.path.join(save_path, f"agentview_{context}_{context_ind[i]}_{j}.png"), img)

                if self.data_aug:
                    agentview_rgb = self.aug(image=agentview_rgb)

                if is_use_hand:
                    eye_in_hand_rgb = obs["robot0_eye_in_hand_image"]
                    state = (agentview_rgb, eye_in_hand_rgb, task_emb)
                else:
                    state = (agentview_rgb, None, task_emb)

                action = agent.predict(state)[0]
                obs, r, done, _ = env.step(action)

                # if self.render:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    # env.close()
                    break

            env.close()

    def test_agent(self, agent, cpu_set=None, epoch=None, is_save=False, folder="", task_suite="", seed=10000, is_osm=False, is_use_hand=True):
        self.is_osm = is_osm
        self.task_suite = task_suite
        self.seed = seed
        logging.info("Start testing agent")

        self.task_embs = agent.trainset.tasks

        if cpu_set is None:
            num_cpu = mp.cpu_count()
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)
            print(f"cpu_set: {cpu_set}")

        print("There is {} cpus".format(num_cpu))

        print(f"self.task_suite: {self.task_suite}")
        if (self.task_suite == "libero_90") \
                or (self.task_suite == "single_step") \
                or (self.task_suite == "multi_step_2") \
                or (self.task_suite == "multi_step_3"):
            num_tasks = 44
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode
        ###################################################################
        # distribute every runs on cpu
        ###################################################################
        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        repeat_num = all_runs // num_cpu
        repeat_res = all_runs % num_cpu

        workload_array = np.ones([num_cpu], dtype=int)
        workload_array[:repeat_res] += repeat_num
        workload_array[repeat_res:] = repeat_num

        assert np.sum(workload_array) == all_runs

        ind_workload = np.cumsum(workload_array)
        ind_workload = np.concatenate([[0], ind_workload])
        ###################################################################
        ctx = mp.get_context('spawn')
        processes_list = []

        # for i in range(self.n_cores):
        for i in range(num_cpu):  # yy: I modified this
            p = ctx.Process(target=self.eval_agent,
                            kwargs={
                                "agent": agent,
                                "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                "success": success,
                                "pid": i,
                                "cpu_set": set(cpu_set[i:i + 1]),
                                "is_osm": self.is_osm,
                                "task_suite_": self.task_suite,
                                "is_use_hand": is_use_hand
                            },
                            )
            p.start()
            processes_list.append(p)

        [p.join() for p in processes_list]

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        print(f'success array {success.detach()}')

        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        if is_save:
            success_rate_npy = success_rate.cpu().numpy()
            np.save(f"{folder}/ori_way_to_eval_succ_list_bm_{task_suite}_seed_{seed}.npy", success_rate_npy)

        print(num_tasks)
        for num in range(num_tasks):
            print(f"Task {num}: {success_rate[num].item()}")
            log.info(f"Task {num}: {success_rate[num].item()}")

            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        log.info(f"Average success rate: {average_success}")
