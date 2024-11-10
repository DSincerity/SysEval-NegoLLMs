import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from typing import List
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix
from baselines.tasks.task_selector import TaskSelector
import numpy as np

class EvaluationMetrics(object):

    def __init__(self):
        pass

    def compute_metric(self, preds:List, gt:List, metric:str):
        if metric == 'accuracy':
            return accuracy_score(gt, preds)
        elif metric == 'f1_per_class':
            return self.f1_per_class_score(gt, preds)
        elif metric == 'elementwise_accuracy':
            return self.elementwise_accuracy_score(gt, preds)
        else:
            raise NotImplementedError(f"{metric} is not implemented")

    def f1_per_class_score(self, gt:List, preds:List):

        mlb = MultiLabelBinarizer()

        mlb.fit(gt + preds)
        label_indices = {index: label for index, label in enumerate(mlb.classes_)}
        ground_truth_binary = mlb.transform(gt)
        predictions_binary = mlb.transform(preds)
        # create a confusion matrix.
        confusion_matrix = multilabel_confusion_matrix(ground_truth_binary, predictions_binary)
        #print('confusion_matrix :', confusion_matrix)

        # calculate metrics.
        # calculate the accuracy and f1 score for each label.
        label_accuracies = {}
        label_f1_scores = {}
        for index in range(confusion_matrix.shape[0]):    # going through each class
            tn = confusion_matrix[index, 0, 0]
            fp = confusion_matrix[index, 0, 1]
            fn = confusion_matrix[index, 1, 0]
            tp = confusion_matrix[index, 1, 1]

            label = label_indices[index]

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            label_accuracies[label] = accuracy

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = (2 * precision * recall) / (precision + recall+ 1e-10)
            label_f1_scores[label] = f1

        # compute average accuracy.
        average_accuracy = sum(label_accuracies.values()) / len(label_accuracies)

        # compute average f1 score.
        average_f1 = sum(label_f1_scores.values()) / len(label_f1_scores)

        return average_f1

    def elementwise_accuracy_score(self, gt:List, preds:List):
        assert len(gt) == len(preds), 'length of ground truth and prediction should be same'

        return np.mean([str(val) == str(_pred.get(item,'None')) for _gt, _pred in zip(gt, preds) for item, val in _gt.items()])

    @staticmethod
    def get_eval_method_by_task():
        evaluation_method =dict()
        task_selector = TaskSelector()
        classfication_tasks= task_selector.get_tasks_by_dataset_type("all", "classification")
        regression_tasks= task_selector.get_tasks_by_dataset_type("all", "regression")
        proposal_tasks= task_selector.get_tasks_by_mid_types("proposal")
        strategy = task_selector.get_tasks_by_mid_types("strategy")
        multi_dialog_act = [t for t in task_selector.get_tasks_by_mid_types("dialog_act") if 'ji' in t or 'dnd' in t]

        evaluation_method['elementwise_accuracy']= proposal_tasks
        evaluation_method['accuracy']= classfication_tasks+regression_tasks
        evaluation_method['f1_per_class'] = strategy + multi_dialog_act

        task_to_metric = {task: metric for metric, tasks in evaluation_method.items() for task in tasks}
        return task_to_metric
