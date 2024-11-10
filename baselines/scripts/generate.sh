#!/bin/bash
experiment_nm=<experiment_name>
checkpoint=<checkpoint_path>
task_nm=<task_name> # task for generation. e.g., mid_ask_high_priority_ca

python3 baselines/main.py   --storage_dir baselines/processed_datasets \
                            --output_dir model_output/$experiment_nm \
                            --model_name  hf_model \
                            --do_generate  \
                            --checkpoint $checkpoint \
                            --base_model_name google/flan-t5-base \
                            --task_name $task_nm
