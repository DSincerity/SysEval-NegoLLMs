"""
Task Question: In the final deal, how many items did you get for each issue? If the agents did not reach a deal, answer 0. (respond as a dict)
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y
"""

from tasks.task_w import WBaseTaskHandler


class BidHandlerJI(WBaseTaskHandler):
    """Handler for the task of determining the results of a Job Interview."""

    possible_keys = ["Company", "Position", "Workplace", "Salary", "Days_off"]

    possible_outputs = ["Google", "Apple", "Facebook", "Amazon", "NA"]
    possible_outputs += ["Engineer", "Manager", "Designer", "Sales"]
    possible_outputs += ["Tokyo", "Seoul", "Beijing", "Sydney"]
    possible_outputs += list(str(x) for x in range(20, 51)) + ["0"]
    possible_outputs += ["2", "3", "4", "5"]

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_dial_template(counts_bool=True, cot_bool=model_handler.cot, values_bool=True, dialogue_bool=False, full_dialogue_bool=True)

        prompt_template = base_template.replace("$question$", "In the final deal, what value was agreed on for each issue?").replace("$output_specification$", "Present your answer as a json within <answer> </answer> tags with keys as issues (Company, Position, Workplace, Salary, Days_off) and values as the corresponding answers. If you are unsure, pick your best guess.")

        return prompt_template

    def evaluate(self, dataset_handler, model_handler,
                         instances, prompts, ground_truth, return_prompt_gt=False):
        """Evaluate the task. Stores the prompts, instances, outputs,
        and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
        """

        # get the model outputs - dict from prompt to the output.
        # It's possible that some are missing so a dict is better than a list.
        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        outputs_dict = model_handler.get_model_outputs(new_prompts, new_ground_truth)

        #only for the ones that are unique and where valid predictions are available
        final_prompts, final_predictions, final_ground_truth = self.get_final_outputs_dict(outputs_dict, self.possible_keys, self.possible_outputs, new_prompts, new_ground_truth)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances


class FComHandler(BidHandlerJI):
    """Handler for the task of checking the final company agreed on
    in a job interview."""

    def generate_prompts(self, dataset_handler, model_handler):
        # get the instances from the dataset
        instances = dataset_handler.get_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model
        prompts = []
        for instance in instances:
            # isolate dialogue from data
            prompt = self.get_prompt_with_bids_ji(instance, prompt_template)

            # prompt = prompt.replace("$issue$", "company")
            # prompt = prompt.replace("$type$", "single word")
            # prompt = prompt.replace("$default$", "None")
            prompts.append(prompt)

        return(prompts, instances)

    def get_ground_truth(self, instances):
        """Get the ground truth for the task.

        Args:
            instances: A dictionary of rows from the dataset.
        """
        ground_truth = []
        for instance in instances:
            gt = {}

            last_bid = instance.bids[-1].options
            if instance.bids[-1].accepted == True:
                gt = {
                    "Company": last_bid['Company'],
                    "Position": last_bid["Position"],
                    "Workplace": last_bid["Workplace"],
                    "Salary": last_bid["Salary"],
                    "Days_off": last_bid["Weekly holiday"],
                }
            else:
                gt = {
                    "Company": "NA",
                    "Position": "NA",
                    "Workplace": "NA",
                    "Salary": "NA",
                    "Days_off": "NA",
                }

            ground_truth.append(gt)
        return ground_truth

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        """Evaluate the task. Stores the prompts, instances, outputs,
        and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
        """
        (prompts, instances) = self.generate_prompts(dataset_handler, model_handler)

        ground_truth = self.get_ground_truth(instances)

        # remove cases where the ground truth is all NA
        assert len(prompts) == len(ground_truth)
        prompts2, ground_truth2 = [], []
        for i in range(len(prompts)):
            # check if all ground truth values are NA
            if not all([x == "NA" for x in ground_truth[i].values()]):
                prompts2.append(prompts[i])
                ground_truth2.append(ground_truth[i])

        prompts, ground_truth = prompts2, ground_truth2

        super().evaluate(dataset_handler, model_handler,
                         instances, prompts, ground_truth)
