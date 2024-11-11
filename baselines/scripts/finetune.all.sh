#!/bin/bash
experiment_nm='train_t5_for_all_task'

python3 baselines/main.py   --storage_dir baselines/processed_datasets \
                            --output_dir model_output/$experiment_nm \
                            --model_name  hf_model \
                            --do_train  \
                            --do_generate \
                            --base_model_name google/flan-t5-base \
                            --task_dataset all \
                            --task_type all \
                            --lr 5e-4 \
                            --train_bs 3 \
                            --eval_bs 3 \
                            --num_epochs 5 \
                            --grad_accum_step 1 \
                            --loging_step 50 \
                            --eval_step 500 \
                            --metric_for_best_model loss \

# --storage_dir: Directory for storing datasets
# --output_dir: Directory for model outputs (uses experiment name variable)
# --model_name: Specify the model name
# --do_train: Enable training
# --do_generate: Enable generation
# --base_model_name: Specify the base model name
# --task_name: Choose the specific task name. If this argument is povided, the task_dataset and task_type will be ignored
# --task_dataset: Choose the task datasets in ["all", "ca", "dnd", "ji", "cra"]. Multiple datasets can be selected by separating them with commas (e.g., "ca,dnd")
# --task_type: Choose the task type ["all", "classification", "regression", "multi_outputs", "generation"]. Multiple types can be selected by separating them with commas (e.g., "classification,regression")
# --lr: Set the learning rate
# --train_bs: Training batch size
# --eval_bs: Evaluation batch size
# --num_epochs: Number of epochs
# --grad_accum_step: Gradient accumulation steps
# --loging_step: Logging interval
# --eval_step: Evaluation interval
# --metric_for_best_model: Metric to determine the best model (loss)
