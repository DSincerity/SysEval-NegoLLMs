#!/bin/bash

# setup
MODELSTR="gpt-4-1106-preview"
NUM_INSTANCES=3

# run
echo ""
echo "Running the model now."
echo ""

# Using partial dialogues
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_ask_low_priority_ca,mid_ask_high_priority_ca,mid_partner_ask_low_priority_ca,mid_partner_ask_high_priority_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_utts_partial_dial 2
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_ask_low_priority_ca,mid_ask_high_priority_ca,mid_partner_ask_low_priority_ca,mid_partner_ask_high_priority_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_utts_partial_dial 4
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_ask_low_priority_ca,mid_ask_high_priority_ca,mid_partner_ask_low_priority_ca,mid_partner_ask_high_priority_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_utts_partial_dial 6
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_ask_low_priority_ca,mid_ask_high_priority_ca,mid_partner_ask_low_priority_ca,mid_partner_ask_high_priority_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_utts_partial_dial 8

# Use Chain of Thought
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "end_deal_total_ca,sta_max_points_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --use_cot True
python main.py --storage_dir storage/ --dataset_name dnd --model_name open_ai --task_name "sta_max_points_dnd,end_deal_total_dnd" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --use_cot True

# Adding prior utterances
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_strategy_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_prior_utts 2
python main.py --storage_dir storage/ --dataset_name dnd --model_name open_ai --task_name "mid_dial_act_dnd" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_prior_utts 2
python main.py --storage_dir storage/ --dataset_name job_interview --model_name open_ai --task_name "mid_dial_act_ji" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_prior_utts 2
python main.py --storage_dir storage/ --dataset_name cra --model_name open_ai --task_name "mid_dial_act_cra" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_prior_utts 2

# Few-shot evaluation
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_strategy_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_multishot 2
python main.py --storage_dir storage/ --dataset_name dnd --model_name open_ai --task_name "mid_dial_act_dnd" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_multishot 2
python main.py --storage_dir storage/ --dataset_name job_interview --model_name open_ai --task_name "mid_dial_act_ji" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_multishot 2
python main.py --storage_dir storage/ --dataset_name cra --model_name open_ai --task_name "mid_dial_act_cra" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR --num_multishot 2

echo "DONE"
