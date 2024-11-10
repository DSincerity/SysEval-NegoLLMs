import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils import json_loader


class TaskSelector:
    """Class to select tasks based on task information from the input arguments."""

    def __init__(self, baseline_type="T5"):
        self.task_class = json_loader("./baselines/tasks/TASKTYPE.json")[baseline_type]

    def get_task_types(self):
        return list(self.task_class.keys())

    def get_mid_types(self):
        return [
            task
            for task_type in self.task_class.keys()
            for task in self.task_class[task_type]
        ]

    def get_tasks_by_mid_types(self, mid_type):
        mid_type = mid_type.split(",")
        assert len([md for md in mid_type if md in self.get_mid_types()]) == len(
            mid_type
        ), "mid_type should be one of {}".format(self.get_mid_types())

        final = []
        for dn in mid_type:
            final.extend(
                [
                    task
                    for mid_task_type in self.task_class.values()
                    if dn in mid_task_type
                    for task in mid_task_type[dn]
                ]
            )
        return final

    def get_all_tasks(self):
        from itertools import chain

        return list(
            chain(
                *[
                    task
                    for task_type in self.task_class.values()
                    for task in task_type.values()
                ]
            )
        )

    def get_tasks_by_dataset(self, dataset_name):
        all_tasks = self.get_all_tasks()
        dataset_name = dataset_name.split(",")
        assert len(
            [dn for dn in dataset_name if dn in ["ca", "dnd", "ji", "cra"]]
        ) == len(dataset_name), "dataset_name should be one of ca, dnd, ji, cra"

        final = []
        for dn in dataset_name:
            final.extend([task for task in all_tasks if task.endswith(dn)])
        return final

    def get_tasks_by_type(self, type_name):
        from itertools import chain

        type_name = type_name.split(",")
        assert len(
            [
                _type
                for _type in type_name
                if _type
                in ["classification", "regression", "multi_outputs", "generation"]
            ]
        ) == len(
            type_name
        ), "type_name should be one of classification, regression, multi_regression"

        final = []
        for task_type in type_name:
            tasks = self.task_class[task_type]
            final.extend(list(chain(*[task for task in tasks.values()])))
        return final

    def get_tasks_by_dataset_type(self, dataset_name, type_name):
        if dataset_name == "all" and type_name == "all":
            return self.get_all_tasks()
        elif dataset_name == "all" and type_name != "all":
            return self.get_tasks_by_type(type_name) or self.get_tasks_by_mid_types
        elif dataset_name != "all" and type_name == "all":
            return self.get_tasks_by_dataset(dataset_name)
        elif dataset_name != "all" and type_name != "all":
            type_name_sp = type_name.split(",")
            dataset_name_sp = dataset_name.split(",")
            if len(type_name_sp) == 1 and dataset_name_sp[0] in self.get_all_tasks():
                return [type_name]
            tasks = (
                self.get_tasks_by_type(type_name)
                if len([tn for tn in type_name_sp if tn in self.get_task_types()])
                == len(type_name_sp)
                else self.get_tasks_by_mid_types(type_name)
            )
            return [task for task in tasks if task.split("_")[-1] in dataset_name_sp]
        else:
            raise NotImplementedError


if __name__ == "__main__":
    task_selector = TaskSelector()
    print("get_task_types: ", task_selector.get_task_types())
    print("get_mid_types: ", task_selector.get_mid_types())
    print(
        "get_tasks_by_dataset_type: all + all\n",
        task_selector.get_tasks_by_dataset_type("all", "all"),
    )
    print(
        "get_tasks_by_dataset_type: ca + classification\n",
        task_selector.get_tasks_by_dataset_type("ca", "classification"),
    )
    print(
        "get_tasks_by_dataset_type: all + generation\n",
        task_selector.get_tasks_by_dataset_type("all", "generation"),
    )
