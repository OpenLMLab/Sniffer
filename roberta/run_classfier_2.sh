#!/usr/bin/env bash

function run_exp() {
  gpu=${1}
  model_name=${2}
  train_data=${3}
  val_data=${4}
  max_len=${5}
  batch_size=${6}
  lr=${7}
  type=${8}
  grad_ac=${9}
  train=${10}
  language=${11}

  exp_name=sniffer_${model_name}_${type}_${language}
  output_dir=outputs/${exp_name}

  # python -m torch.distributed.launch --nproc_per_node 4 classifier.py \
  # python classifier.py \

  WANDB_PROJECT=sniffer CUDA_VISIBLE_DEVICES=${gpu} python classifier.py \
    --model_name_or_path ${model_name} \
    --train_file ${train_data} \
    --validation_file ${val_data} \
    --do_train ${train}\
    --do_eval True\
    --pad_to_max_length False \
    --gradient_accumulation_steps ${grad_ac} \
    --max_seq_length ${max_len} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 128\
    --learning_rate ${lr} \
    --num_train_epochs 5 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --logging_steps 100 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --run_name ${exp_name} \
    --output_dir ${output_dir} \
    --language ${language} \
    --overwrite_output_dir
}

#run_exp    gpu         model_name           train_data                    val_data           max_len   batch_size    lr        type             grad_ac    do_train   language
#run_exp      1        roberta-large      data/en_data/train.json    data/en_data/test.json       256       16         2e-5     origin_tracing       1         True       en
# run_exp      1        roberta-large      data/cn_data/train.json    data/cn_data/test.json       256       16         2e-5     origin_tracing       1         True       cn

# run_exp      6        roberta-base      data/raw_processed_features/en_train_raw_1pct.json    data/raw_processed_features/en_test_raw.json        1024       4         1e-5     few_shot_en_raw_1pct       1         True       en
# run_exp      6        roberta-base      data/raw_processed_features/en_train_raw_10pct.json    data/raw_processed_features/en_test_raw.json       1024       8         2e-5     few_shot_en_raw_10pct       1         True       en
run_exp      6        roberta-base      data/raw_processed_features/en_train_raw.json    data/raw_processed_features/en_test_raw.json             1024       16         3e-5     few_shot_en_raw_100pct       1         True       en