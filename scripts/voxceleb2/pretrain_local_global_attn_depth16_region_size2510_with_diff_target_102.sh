# Set the path to save checkpoints
server=102
model="videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target"
OUTPUT_DIR="./saved/model/pretraining/voxceleb2/${model}_server${server}"
if [ ! -d "$OUTPUT_DIR" ]; then
mkdir -p $OUTPUT_DIR
fi
# Set the path to pre-training dataset.
DATA_PATH='./saved/data/voxceleb2/info_clean.csv'
# batch_size can be adjusted according to number of GPUs
# this script is for 4 GPUs (1 nodes x 4 GPUs)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 17320 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type part_window \
        --mask_ratio 0.9 \
        --input_size 160 \
        --model pretrain_videomae_base_dim512_no_depth_patch16_160 \
        --encoder_depth 16 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 10 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 20 \
        --attn_type local_global \
        --part_win_size 2 5 10 \
        --lg_region_size 2 5 10 \
        --use_frame_diff_as_target \
#        >${OUTPUT_DIR}/nohup.out 2>&1 &
