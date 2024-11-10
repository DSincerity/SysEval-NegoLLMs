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

    parser.add_argument(
        "--task_name",
        type=str,
        default="sta_total_item_count_dnd",
        help="the task name. use commas to separate a list of tasks.",
    )

    # nego_datasets
    parser.add_argument(
        "--num_instances",
        type=int,
        default=10,
    )

    # max cutoff for the num instances - regardless of what the above param says.
    parser.add_argument(
        "--max_num_instances",
        type=int,
        default=200,
    )

    # num utterances for the partial dialogue (used for tasks like mid_ask_low_priority_ca and mid_partner_ask_low_priority_ca). This includes utterances from both agents. So if you want 2 utterances from each agent, set this to 4.
    parser.add_argument(
        "--num_utts_partial_dial",
        type=int,
        default=-1,  # -1 means all utterances
    )

    # chain of thought
    parser.add_argument(
        "--use_cot",
        type=bool,
        default=False,
    )

    # multishot
    parser.add_argument(
        "--num_multishot",
        type=int,
        default=0,  # 0 means only the utterance that needs to be annotated will be used.
    )

    # prior context
    parser.add_argument(
        "--num_prior_utts",
        type=int,
        default=0,  # 0 means only the utterance that needs to be annotated will be used without any prior context
    )

    # hf_model huggingface string name as hosted in the Huggingface hub here: https://huggingface.co/models
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
