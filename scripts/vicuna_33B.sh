#!/bin/bash

# setup
#export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
MODELSTR="lmsys/vicuna-33b-v1.3"
NUM_INSTANCES=1

# run
echo ""
echo "Running the model now."
echo ""

# different test cased based on the dataset.
python main.py --storage_dir storage/ --dataset_name dnd --model_name hf_model --task_name "sta_total_item_count_dnd,sta_max_points_dnd,mid_dial_act_dnd,mid_gen_resp_dnd,end_deal_specifics_dnd,sta_ask_point_values_dnd,mid_full_proposal_dnd,end_deal_total_dnd" --num_instances $NUM_INSTANCES --hf_model_str $MODELSTR
python main.py --storage_dir storage/ --dataset_name casino --model_name hf_model --task_name "mid_strategy_ca,mid_gen_resp_ca,end_deal_specifics_ca,end_deal_total_ca,sta_max_points_ca,sta_ask_point_values_ca,sta_ask_low_priority_ca,sta_ask_high_priority_ca,mid_ask_low_priority_ca,mid_ask_high_priority_ca,mid_partner_ask_low_priority_ca,mid_partner_ask_high_priority_ca,end_deal_likeness_ca,end_deal_satisfaction_ca,end_partner_deal_likeness_ca,end_partner_deal_satisfaction_ca,sta_total_item_count_ca" --num_instances $NUM_INSTANCES --hf_model_str $MODELSTR
python main.py --storage_dir storage/ --dataset_name job_interview --model_name hf_model --task_name "end_deal_specifics_ji,sta_ask_high_priority_ji_w,sta_ask_low_priority_ji_w,mid_ask_high_priority_ji_w,mid_ask_low_priority_ji_w,mid_partner_ask_high_priority_ji_w,mid_partner_ask_low_priority_ji_w,mid_dial_act_ji" --num_instances $NUM_INSTANCES --hf_model_str $MODELSTR
python main.py --storage_dir storage/ --dataset_name cra --model_name hf_model --task_name "mid_dial_act_cra,mid_full_proposal_cra" --num_instances $NUM_INSTANCES --hf_model_str $MODELSTR

echo "DONE"
