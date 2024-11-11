"""
Task Question: In the final deal, how many items did you get for each issue? If the agents did not reach a deal, answer 0. (respond as a dict)
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y
"""


from tasks.task_w import WBaseTaskHandler


class CHandlerDND(WBaseTaskHandler):
    """Handler for the task of checking the counts of
    items that each person in the negotiation recieved."""

    # for every item in the json
    possible_keys = ["books", "hats", "balls"]
    possible_outputs = [str(x) for x in range(-40, 40)]
    possible_outputs.append("NA")

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, dialogue_bool=False, da_bool=False, cot_bool=model_handler.cot, full_dialogue_bool=True)

        prompt_template = base_template.replace("$question$", "In the final deal, how many items of each issue did you get?").replace("$output_specification$", "Present your answer as a json within <answer> </answer> tags with keys as issues (books, hats, and balls) and values as the corresponding answers. If you are unsure, pick your best guess.")

        return prompt_template

    def evaluate(self, dataset_handler, model_handler, instances, prompts, ground_truth):
        """Evaluate the task. Stores the prompts, instances, outputs,
        and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
            instances: The instances from the dataset.
            prompts: The prompts for the task.
            ground_truth: The ground truth for the task.
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

class A1BCHandler(CHandlerDND):
    """Handler for the task of checking the total books that YOU recieved."""

    def generate_prompts(self, dataset_handler, model_handler):
        # get the instances from the dataset
        instances = dataset_handler.get_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model
        prompts = []
        for instance in instances:
            # isolate dialogue from data
            prompt = self.get_prompt_dnd(instance, prompt_template, "YOU")

            # dict format only from you perspective. - not required.
            # prompt = prompt.replace("$agent_name$", "Agent 1")
            # prompt = prompt.replace("$item$", "books")
            prompts.append(prompt)

        return(prompts, instances)

    def get_ground_truth(self, instances):
        """Get the ground truth for the task.

        Args:
            instances: A dictionary of rows from the dataset.
        """
        ground_truth = []
        for instance in instances:
            if "no_agreement" in instance["output"]:
                # just fill with NA
                gt = {
                    "books": "NA",
                    "hats": "NA",
                    "balls": "NA"
                }
            else:
                gt = {
                    "books": instance['output'].split()[0].split("=")[-1],
                    "hats": instance['output'].split()[1].split("=")[-1],
                    "balls": instance['output'].split()[2].split("=")[-1]
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

        super().evaluate(dataset_handler, model_handler,
                         instances, prompts, ground_truth)
