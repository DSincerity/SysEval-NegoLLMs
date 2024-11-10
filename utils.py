import importlib
import json
import os
import csv

from registry import CLS_NAME2PATHS


def get_dataset_handler(dataset_name, args):
    """Get the dataset handler."""

    # assisted by ChatGPT
    class_name = CLS_NAME2PATHS["nego_datasets"][dataset_name]
    module_name, _, class_name = class_name.rpartition(".")
    module = importlib.import_module(module_name)
    class_to_use = getattr(module, class_name)
    dataset_handler = class_to_use(dataset_name, args)

    return dataset_handler


def get_model_handler(model_name, args):
    """Get the model handler."""

    # assisted by ChatGPT
    class_name = CLS_NAME2PATHS["models"][model_name]
    module_name, _, class_name = class_name.rpartition(".")
    module = importlib.import_module(module_name)
    class_to_use = getattr(module, class_name)
    model_handler = class_to_use(model_name, args)

    return model_handler


def get_task_handler(task_name, args):
    """Get the task handler."""

    # assisted by ChatGPT
    class_name = CLS_NAME2PATHS["tasks"][task_name]
    module_name, _, class_name = class_name.rpartition(".")
    module = importlib.import_module(module_name)
    class_to_use = getattr(module, class_name)
    task_handler = class_to_use(task_name, args)

    return task_handler


def get_output_path(
    storage_dir, dataset_name, model_name, task_name, num_instances, args=None
):
    """Construct the output path."""

    assert args != None
    assert not isinstance(args, int)

    fname = f"{dataset_name}_{model_name}_{task_name}_{num_instances}"

    if args.num_utts_partial_dial != -1:
        fname += f"_partial_{args.num_utts_partial_dial}"

    if args.use_cot:
        fname += "_cot"

    if args.num_multishot != 0:
        fname += f"_multishot_{args.num_multishot}"

    if args.num_prior_utts != 0:
        fname += f"_prior_utts_{args.num_prior_utts}"

    fname = f"{fname}.json"

    return os.path.join(storage_dir, "logs", fname)


def get_connection_info_path(storage_dir):
    """Construct the path to the connection info for the models."""

    return os.path.join(storage_dir, "utilities", "connection_info.json")


def json_loader(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json(data, file_path):
    """Save the data in a json file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def write_to_csv(data, file_path):
    assert isinstance(data, list), "data must be a list of strings"
    with open(file_path, "w") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        # write.writerow(fields)
        write.writerows(data)
