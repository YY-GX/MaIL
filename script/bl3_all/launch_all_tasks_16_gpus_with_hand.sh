#!/bin/zsh

# Loop through task_order_index from 27 to 42
for task_order_index in {27..42}; do
    # Submit the Slurm job
    sbatch --cpus-per-task=8 --gpus=1 -o "${ENDPOINT}/pkgs_baselines/MaIL/outs/train_mail_bl3_all_${task_order_index}_%j.out" -J "mail_train_${task_order_index}" --wrap="\
    python run_benchmark_separate.py --config-name=benchmark_libero_separate \
        --multirun agents=goal_ddpm_mamba_encdec_separate \
        agent_name=ddpm_mamba_cross \
        folder_name=bl3_all_1000_epoch_with_hand \
        epoch=1000 \
        is_bl3_all=True \
        task_suite=libero_90 \
        task_order_index=${task_order_index} \
        group=libero_90_ddpm_mamba_cross_goal \
        obs_seq=5 \
        train_batch_size=128 \
        n_layer_encoder=8 \
        mamba_encoder_cfg.d_state=8 \
        mamba_decoder_cfg.d_state=8 \
        enc_conv=2 \
        dec_conv=2 \
        seed=10000 \
        wandb.entity=yygx \
        wandb.project=mail"

    # Print a confirmation message
    echo "Submitted task_order_index=${task_order_index}"

    # Wait for 2 seconds before the next iteration
    sleep 2
done


#Submitted batch job 8219
#Submitted task_order_index=27
#Submitted batch job 8220
#Submitted task_order_index=28
#Submitted batch job 8221
#Submitted task_order_index=29
#Submitted batch job 8222
#Submitted task_order_index=30
#Submitted batch job 8223
#Submitted task_order_index=31
#Submitted batch job 8224
#Submitted task_order_index=32
#Submitted batch job 8225
#Submitted task_order_index=33
#Submitted batch job 8226
#Submitted task_order_index=34
#Submitted batch job 8227
#Submitted task_order_index=35
#Submitted batch job 8228
#Submitted task_order_index=36
#Submitted batch job 8229
#Submitted task_order_index=37
#Submitted batch job 8230
#Submitted task_order_index=38
#Submitted batch job 8231
#Submitted task_order_index=39
#Submitted batch job 8232
#Submitted task_order_index=40
#Submitted batch job 8233
#Submitted task_order_index=41
#Submitted batch job 8234
#Submitted task_order_index=42