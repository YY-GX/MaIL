python run_benchmark.py  --config-name=benchmark_libero_goal \
            --multirun agents=goal_mamba_only_agent \
            agent_name=ddpm_mamba \
            task_suite=libero_90,libero_goal \
            wandb.project=ablation \
            group=video \
            train_batch_size=128 \
            n_layers=16 \
            seed=2