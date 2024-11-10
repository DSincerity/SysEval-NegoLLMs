import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tqdm import tqdm
from utils import json_loader, write_json
from collections import Counter
from baselines.metrics.eval_metric import EvaluationMetrics


class MajorityBaseline(object):

    def __init__(self, result_path, report_save_path):
        self.whole_df = json_loader(result_path)
        self.task_majority_preds_gt = dict()
        self.final_report = dict()
        self.report_save_path=report_save_path
        self.task_to_metric = EvaluationMetrics().get_eval_method_by_task()

    def get_majority(self):

        for task, data in tqdm(self.whole_df.items()):
            train_gt = data['train'][1]
            test_gt = data['test'][1]
            test_size = len(test_gt)

            # get majority value
            majority_value, cnt = Counter(train_gt).most_common()[0]

            # preprocessing dataset by eval method per task
            if self.task_to_metric[task] == 'f1_per_class':
                majority_value = majority_value.split(', ')
                test_gt = [gt.split(', ') for gt in test_gt]
                assert type(test_gt[0]) and type(majority_value), 'ground truth and majority value are expected to be list'

            elif self.task_to_metric[task] == 'elementwise_accuracy':
                assert isinstance(majority_value, str), 'majority value is expected to be string'
                majority_value = json.loads(majority_value)
                test_gt = [json.loads(gt) for gt in test_gt]
                assert type(test_gt[0]) and type(majority_value), 'ground truth and majority value are expected to be dict'

            elif self.task_to_metric[task] == 'accuracy':
                assert type(test_gt[0]) and type(majority_value), 'ground truth and majority value are expected to be string'
                pass

            print(f"[{task} - {self.task_to_metric[task]} ] ground truth ({test_gt[0]}) vs majority prediction({majority_value})")

            preds = [majority_value] * test_size
            self.task_majority_preds_gt[task]=dict()
            self.task_majority_preds_gt[task]['preds'] = preds
            self.task_majority_preds_gt[task]['gt'] = test_gt
            #print(f"[{task}] {majority_value}, {cnt}")

        return self.task_majority_preds_gt

    def evaluation(self):
        evaluation = EvaluationMetrics()
        for task, result in self.task_majority_preds_gt.items():
            pred, gt = result['preds'], result['gt']
            eval_metric = self.task_to_metric[task]
            self.final_report[task]= evaluation.compute_metric(preds=pred, gt=gt, metric=eval_metric)
        return self.final_report


if __name__ == "__main__":
    result_path = "<result_path>"
    report_save_path = "<result_save_path>"
    MB= MajorityBaseline(result_path=result_path, report_save_path=report_save_path)
    MB.get_majority()
    final_report = MB.evaluation()
    write_json(final_report, MB.report_save_path)
