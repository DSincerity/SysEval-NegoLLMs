"""
File to manage all arguments for the repository.

Group all arguments, as required for better organization. For instance, keeping all arguments related to a specific model or test together.
"""


from argparse import ArgumentParser


def add_arguments(parent_parser):
    """Specify all the arguments - group them as required."""
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # I/O
    parser.add_argument(
        "--storage_dir",
        type=str,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dnd",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="llama",
    )

    parser.add_argument("--task_name", type=str, default=None)

    # nego_datasets
    parser.add_argument(
        "--num_instances",
        type=int,
        default=10,
    )

    # num utterances for the partial dialogue (used for tasks like mid_ask_low_priority_ca and mid_partner_ask_low_priority_ca). This could includes utterances from both agents. So if you want 2 utterances from each agent, set this to 4.
    parser.add_argument(
        "--num_utts_partial_dial",
        type=int,
        default=-1,  # -1 means all utterances
    )

    parser.add_argument(
        "--hf_model_str",
        type=str,
        default="google/flan-t5-small",
    )

    parser.add_argument(
        "--openai_model_str",
        type=str,
        default="gpt-3.5-turbo-0613",
    )
    return parser


def add_model_arguments(parent_parser):
    """Specify all the arguments - group them as required."""
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument(
        "--seed", type=int, help="seed for random number generator", default=1234
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint to be loaded.",
        default=None,
    )

    parser.add_argument(
        "--base_model",
        type=str,
        help="name of the base model. defulat value: google/flan-t5-base",
        default="google/flan-t5-base",
    )
    parser.add_argument(
        "--setup_dataset",
        help="whether to call setup function in dataset handler",
        action="store_true",
    )

    parser.add_argument(
        "--overall_eval",
        help="whether to evaluate/geneartion overall dataset or each tasks",
        action="store_true",
    )

    parser.add_argument(
        "--do_train",
        help="whether to train the model",
        action="store_true",
    )

    parser.add_argument(
        "--do_eval",
        help="whether to evaluate the model with tasks",
        action="store_true",
    )

    parser.add_argument(
        "--do_generate",
        help="whether to generate ouput with the trained model",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
        default="./",
    )
    parser.add_argument(
        "--task_dataset",
        type=str,
        help="name of the task dataset. defulat value: all",
        default="all",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        help="name of the task type. classification/regression/multi_outputs/generation. defulat value: all",
        default="all",
    )

    parser.add_argument(
        "--base_model_name",
        help="name of the base model. defulat value: google/flan-t5-base",
        default="google/flan-t5-base",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        help="checkpoint path to resume from checkpoint",
        action="store_true",
    )

    parser.add_argument(
        "--train_file",
        help="train file path",
        default=None,
    )

    parser.add_argument(
        "--test_file",
        help="test file path",
        default=None,
    )

    return parser


def add_model_hyperparameters(parent_parser):
    """Specify all the arguments - group them as required."""
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=5e-4,
    )

    parser.add_argument(
        "--train_bs",
        type=int,
        help="train batch size",
        default=16,
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        help="path to tokenizer",
        default=None,
    )

    parser.add_argument(
        "--eval_bs",
        type=int,
        help="eval batch size",
        default=16,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="number of epochs",
        default=5,
    )

    parser.add_argument(
        "--grad_accum_step", type=int, help="gradient accumulation step", default=1
    )

    parser.add_argument("--loging_step", type=int, help="logging step", default=50)

    parser.add_argument("--eval_step", type=int, help="eval step", default=200)

    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        help="metric for best model",
        default="loss",
    )

    parser.add_argument(
        "--greater_is_better",
        help="whether greater is better for the metric",
        action="store_true",
    )

    parser.add_argument(
        "--save_whole_dataset",
        help="whether to save the whole dataset for input to the model",
        action="store_true",
    )
    parser.add_argument("--weight_decay", help="weight decay", default=0.01, type=float)

    parser.add_argument("--peft_method", default=None)
    parser.add_argument("--lora_r", default=32, type=int)
    parser.add_argument("--prefix_tokens", default=20, type=int)
    parser.add_argument("--prefix_projection", default=1, type=int)
    parser.add_argument("--lora_dropout", default=0.3, type=float)
    parser.add_argument("--p_tokens", default=20, type=int)
    parser.add_argument("--p_hidden", default=100, type=int)
    parser.add_argument("--prompt_tokens", default=20, type=int)

    return parser
