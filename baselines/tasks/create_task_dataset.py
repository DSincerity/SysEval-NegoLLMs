import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import utils
import json


class CreateDatasetByTasks(object):
    """Class for creating dataset by tasks"""

    def __init__(self, args, task_name, model_name, dataset_name, orgin_dataset):
        self.args = args
        self.task_name = task_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_handler = utils.get_dataset_handler(self.dataset_name, self.args)
        self.model_handler = utils.get_model_handler(self.model_name, self.args)
        self.task_handler = utils.get_task_handler(self.task_name, self.args)

        # Train
        self.dataset_handler.dataset = orgin_dataset[dataset_name]["train"]
        self.dataset_handler.args.num_instances = len(
            orgin_dataset[dataset_name]["train"]
        )
        if dataset_name == "job_interview":
            self.dataset_handler.da_list = orgin_dataset["job_interview_dialacts"][
                "train"
            ]
        elif dataset_name == "dnd":
            self.dataset_handler.annotated_dataset = orgin_dataset["dnd_dialacts"][
                "train"
            ]
        self.train_input_text, self.train_labels = self.task_handler.evaluate(
            self.dataset_handler, self.model_handler, return_prompt_gt=True
        )

        # label processing by task types
        if isinstance(self.train_labels[0], list):
            self.train_labels = [
                ", ".join(sorted(label)) for label in self.train_labels
            ]
        elif isinstance(self.train_labels[0], dict):
            # self.train_labels = [json.dumps(label) for label in self.train_labels]
            self.train_labels = [
                json.dumps(dict(sorted(label.items(), key=lambda x: x[0])))
                for label in self.train_labels
            ]  # sort dict by key alphabetically

        # Test
        self.dataset_handler.dataset = orgin_dataset[dataset_name]["test"]
        self.dataset_handler.args.num_instances = len(
            orgin_dataset[dataset_name]["test"]
        )
        if dataset_name == "job_interview":
            self.dataset_handler.da_list = orgin_dataset["job_interview_dialacts"][
                "test"
            ]
        elif dataset_name == "dnd":
            self.dataset_handler.annotated_dataset = orgin_dataset["dnd_dialacts"][
                "test"
            ]
        self.test_input_text, self.test_labels = self.task_handler.evaluate(
            self.dataset_handler, self.model_handler, return_prompt_gt=True
        )

        # label processing by task types
        if isinstance(self.test_labels[0], list):
            self.test_labels = [", ".join(sorted(label)) for label in self.test_labels]
        elif isinstance(self.test_labels[0], dict):
            self.test_labels = [json.dumps(label) for label in self.test_labels]

    def get_max_len(self, tokenizer):
        max_source_length = max(
            [len(tokenizer(prompt)["input_ids"]) for prompt in self.train_input_text]
        )
        max_label_length = max(
            [len(tokenizer(label)["input_ids"]) for label in self.train_labels]
        )
        return max_source_length, max_label_length


def DummyModel(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
