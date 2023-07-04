server=164
pretrain_dataset='voxceleb2'
pretrain_server=170
finetune_dataset='ferv39k'
num_labels=7
ckpts=(49)
input_size=160
sr=1
model="videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target"
model_dir="${model}_server${pretrain_server}"

lr=1e-3
epochs=100
lg_classify_token_types=('region')
for ckpt in "${ckpts[@]}";
do
  for lg_classify_token_type in "${lg_classify_token_types[@]}";
  do
    OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_lr_${lr}_epoch_${epochs}_size${input_size}_sr${sr}_classify_token_type_${lg_classify_token_type}_server${server}"
    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p $OUTPUT_DIR
    fi
    # path to split files (train.csv/val.csv/test.csv)
    DATA_PATH="./saved/data/${finetune_dataset}/all_scenes"
    # path to pre-trained model
    MODEL_PATH="./saved/model/pretraining/${pretrain_dataset}/${model_dir}/checkpoint-${ckpt}.pth"

    # batch_size can be adjusted according to number of GPUs
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.launch --nproc_per_node=3 \
        --master_port 12356 \
        run_class_finetuning.py \
        --model vit_base_dim512_no_depth_patch16_${input_size} \
        --depth 16 \
        --data_set FERV39k \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 40 \
        --num_sample 1 \
        --input_size ${input_size} \
        --short_side_size ${input_size} \
        --save_ckpt_freq 1000 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr ${lr} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs ${epochs} \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --attn_type local_global \
        --lg_region_size 2 5 10 \
        --lg_classify_token_type ${lg_classify_token_type} \
        --num_workers 16 \
        >${OUTPUT_DIR}/nohup.out 2>&1
  done
done
echo "Done!"

