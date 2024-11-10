import os
import json
import ast
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tqdm import tqdm
from baselines.evaluation.eval_metric import EvaluationMetrics
from utils import json_loader


class T5Baseline(object):
    """Class to get a report evaluatation of T5 baseline model with the given generation results."""

    def __init__(self, data_path, result_path, report_save_path, task_to_metric):
        self.whole_df = json_loader(data_path)
        self.gen_result = json_loader(result_path)
        self.task_preds_gt = dict()
        self.final_report = dict()
        self.report_save_path = report_save_path
        self.task_to_metric = task_to_metric
        self.processing_results()

    def processing_results(self):
        for task, data in tqdm(self.whole_df.items()):
            if task not in self.task_to_metric.keys():
                print(
                    f"the task {task} is not on the list of target tasks for T5baseline"
                )
                continue

            if task not in self.gen_result.keys():
                print(f"the task {task} is not on the list of generation results")
                continue

            print("*" * 10, task, "*" * 10)
            train_gt = data["train"][1]
            test_gt = data["test"][1]
            test_size = len(test_gt)

            # preprocessing dataset by eval method per task
            try:
                if self.task_to_metric[task] == "f1_per_class":
                    results = json.loads(
                        self.gen_result[task]["generations"].replace("'", '"')
                    )
                    preds = []
                    for x in results:
                        preds.append([k.strip() for k in x.split(",")])
                    test_gt = [gt.split(", ") for gt in test_gt]
                    assert type(test_gt[0]) and type(
                        preds[0]
                    ), "ground truth and majority value are expected to be list"

                elif self.task_to_metric[task] == "elementwise_accuracy":
                    preds = json.loads(
                        self.gen_result[task]["generations"]
                        .replace("'{", "{")
                        .replace("}'", "}")
                    )
                    test_gt = [json.loads(gt) for gt in test_gt]
                    assert type(test_gt[0]) and type(
                        preds[0]
                    ), "ground truth and majority value are expected to be dict"

                elif self.task_to_metric[task] == "accuracy":
                    preds = json.loads(
                        self.gen_result[task]["generations"].replace("'", '"')
                    )  # convert generations to list
                    assert (
                        len(preds) == test_size
                    ), "ground truth and majority values should have same length"
                    pass

                elif self.task_to_metric[task] == "bleu_rouge":
                    preds = ast.literal_eval(self.gen_result[task]["generations"])
                    preds = [pred.replace("YOU: ", "") for pred in preds]
                    test_gt = [gt.replace("YOU: ", "") for gt in test_gt]
                    assert (
                        len(preds) == test_size
                    ), "ground truth and majority values should have same length"
                    pass
            except Exception as e:
                print(f"[{task}] Error: {e} => Pass this task to evaluate")
                continue

            self.task_preds_gt[task] = dict()
            self.task_preds_gt[task]["preds"] = preds
            self.task_preds_gt[task]["gt"] = test_gt

        return self.task_preds_gt

    def evaluation(self):
        print("Evaluation start ..")
        evaluation = EvaluationMetrics()
        for task, result in self.task_preds_gt.items():
            print(task)
            pred, gt = result["preds"], result["gt"]
            eval_metric = self.task_to_metric[task]
            self.final_report[task] = evaluation.compute_metric(
                preds=pred, gt=gt, metric=eval_metric
            )
        return self.final_report
