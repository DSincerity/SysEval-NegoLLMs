# setup
MODELSTR="gpt-4-1106-preview"
NUM_INSTANCES=2

# run
echo ""
echo "Running the model for a test only with 2 instances"
echo ""

python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --task_name "mid_ask_low_priority_ca" --num_instances $NUM_INSTANCES --openai_model_str $MODELSTR


