python run_benchmark.py  --config-name=benchmark_libero10 \
            --multirun agents=oc_ddpm_mamba_cross \
            agent_name=ddpm_mamba_cross \
            group=ddpm_mamba_cross_30data \
            obs_seq=1,5 \
            n_layer_encoder=8 \
            n_layer_decoder=10 \
            mamba_encoder_cfg.d_state=8 \
            mamba_decoder_cfg.d_state=8 \
            enc_conv=2 \
            dec_conv=2 \
            num_data=30,50 \
            seed=0,1,2