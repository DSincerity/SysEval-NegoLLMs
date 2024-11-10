"""
Main entry point for the LLM evaluation framework.

Takes in the arguments, evaluates the model, and saves the quantitative results and outputs.
"""


from argparse import ArgumentParser
import os
from registry import SUPPORTED_CONFIGS
import utils
import arguments


def validate_args(args):
    """Validate if the input arguments are valid.

    Things to validate include:
        - If the combination of the model, dataset, and task is supported.
        - If the storage directory exists.
        - If the storage does not already consist the output file for this requested configuration.
    """

    # check if the combination of the model, dataset, and task is supported
    task_names = []
    if "," in args.task_name:
        task_names = args.task_name.split(",")
    else:
        task_names = [args.task_name]

    for tname in task_names:
        if (args.dataset_name, args.model_name, tname) not in SUPPORTED_CONFIGS:
            raise ValueError(
                f"Unsupported configuration: {args.dataset_name}, {args.model_name}, {args.task_name}"
            )

    # check if the storage dir exists
    if not os.path.exists(args.storage_dir):
        raise ValueError(f"Storage directory {args.storage_dir} does not exist.")

    # check if the storage dir does not already consist the output file for this requested configuration
    for tname in task_names:
        mname = (
            args.model_name
            if args.model_name != "hf_model"
            else args.hf_model_str.replace("/", "_")
        )
        out_path = utils.get_output_path(
            args.storage_dir,
            args.dataset_name,
            mname,
            args.task_name,
            args.num_instances,
            args=args,
        )
        if os.path.exists(out_path):
            raise ValueError(f"Output file {out_path} already exists.")

    # print the input arguments in a nice format
    print("Input arguments look good:")
    print("-" * 10)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 10)


def main(args):
    """Driver function."""

    # initialize the dataset
    dataset_handler = utils.get_dataset_handler(args.dataset_name, args)

    # initialize the model
    model_handler = utils.get_model_handler(args.model_name, args)

    task_names = []
    if "," in args.task_name:
        task_names = args.task_name.split(",")
    else:
        task_names = [args.task_name]

    # handle all the tasks
    for tix, tname in enumerate(task_names):
        print(f"{tix}/{len(task_names)}: {tname}")

        # initialize the task
        task_handler = utils.get_task_handler(tname, args)

        # run the task, print aggregate results, and store the results and outputs
        task_handler.evaluate(dataset_handler, model_handler)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = arguments.add_arguments(parser)

    args = parser.parse_args()

    # validate the arguments
    validate_args(args)

    main(args)
