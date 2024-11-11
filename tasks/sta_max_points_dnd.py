"""
Task Question: What is the maximum possible points that you can get in any deal?

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: N
"""


from tasks.task_w import WBaseTaskHandler


class MPNDHandlerDND(WBaseTaskHandler):
    """Handler for the MP task of determining the max number of possible points."""

    possible_outputs = [str(x) for x in range(-40, 40)]

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, dialogue_bool=False, da_bool=False, cot_bool=model_handler.args.use_cot)

        prompt_template = base_template.replace("$question$", "What is the maximum number of points that you can possibly get in any deal?").replace("$output_specification$", "Present your answer as a single number with no additional text.")

        return prompt_template

    def evaluate(self, dataset_handler, model_handler, instances, prompts, ground_truth, return_prompt_gt=False):
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
        final_prompts, final_predictions, final_ground_truth = self.get_final_outputs(outputs_dict, self.possible_outputs, new_prompts, new_ground_truth, cot_bool=model_handler.args.use_cot)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances


class A1MPNDHandlerDND(MPNDHandlerDND):
    """Handler for the task of checking the possible total points YOU
    could recieve."""

    def generate_prompts(self, dataset_handler, model_handler):
        # get the instances from the dataset
        instances = dataset_handler.get_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model and hard code ground truth
        prompts = []
        for instance in instances:
            prompt = self.get_prompt_dnd(instance, prompt_template, "YOU")
            #.replace("$agent_name$", "Agent 1") - not required

            # the part below is already done in get_prompt_dnd.
            # you_value = instance['input']['value']
            # books = you_value[0]
            # hats = you_value[1]
            # balls = you_value[2]

            # prompt = prompt.replace("$book_points$", str(books))
            # prompt = prompt.replace("$hat_points$", str(hats))
            # prompt = prompt.replace("$ball_points$", str(balls))
            prompts.append(prompt)

        return(prompts, instances)

    def get_ground_truth(self, instances):
        """Get the ground truth for the task.

        Args:
            instances: A dictionary of rows from the dataset.
        """
        ground_truth = []
        for _ in instances:
            ground_truth.append("10")

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
