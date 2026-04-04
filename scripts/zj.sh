# This script is used to train TabICL for the second stage of the curriculum learning

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun  --standalone --nproc_per_node=8 ./src/tabicl/train/run.py \
            --wandb_log False \
            --wandb_project TabICL \
            --wandb_name Stage2 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype bfloat16 \
            --np_seed 43 \
            --torch_seed 42 \
            --max_steps 160000 \
            --batch_size 256 \
            --micro_batch_size 4 \
            --lr 2e-4 \
            --optimizer muon \
            --scheduler cosine_warmup \
            --cosine_lr_end 2e-5 \
            --warmup_proportion 0.05 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 4 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 2048 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 5000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ./stage1_compile_tf32/checkpoint/dir  \
            --save_temp_every 50 \
            --save_perm_every 5000 \
            --only_load_model True \
            --freeze_col False \
            --freeze_row False