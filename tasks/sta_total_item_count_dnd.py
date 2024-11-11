"""
Task Question: What is the total number of items being negotiated over?

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: N
"""


from tasks.task_w import WBaseTaskHandler


class TICNDHandlerDND(WBaseTaskHandler):
    """Handler for the TIC task of counting total number of items on the table."""

    possible_outputs = [str(x) for x in range(-40, 40)]

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, dialogue_bool=False, da_bool=False, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "What is the total number of items being negotiated over?").replace("$output_specification$", "Present your answer as a single number with no additional text.")

        return prompt_template

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        """Evaluate the task. Stores the prompts, instances, outputs,
        and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
        """

        # get the instances from the dataset
        instances = dataset_handler.get_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model and get ground_truth
        prompts = []
        ground_truth = []
        for instance in instances:
            prompt = self.get_prompt_dnd(instance, prompt_template, "YOU")
            prompts.append(prompt)

            total_count = sum(instance['input']['count'])
            ground_truth.append(str(total_count))

        # get the model outputs - dict from prompt to the output.
        # It's possible that some are missing so a dict is better than a list.
        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        outputs_dict = model_handler.get_model_outputs(new_prompts, new_ground_truth)

        #only for the ones that are unique and where valid predictions are available
        final_prompts, final_predictions, final_ground_truth = self.get_final_outputs(outputs_dict, self.possible_outputs, new_prompts, new_ground_truth)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances
