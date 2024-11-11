"""
Task Question: How many total points did you get at the end of the negotiation?
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y
"""


from tasks.task_w import WBaseTaskHandler
PRIORITY = {"Low": 3, "Medium": 4, "High": 5}


class PHandlerCa(WBaseTaskHandler):
    """Handler for the task of determining the total points each agent recieved."""

    possible_outputs = [str(x) for x in range(37)]

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, utterance_bool=False, dialogue_bool=False, cot_bool=model_handler.args.use_cot, full_dialogue_bool=True)

        prompt_template = base_template.replace("$question$", "How many points did you get at the end of the negotiation?").replace("$output_specification$", "Present your answer as a single number with no additional text.")

        return prompt_template

    def evaluate(self, dataset_handler, model_handler,
                 instances, prompts, ground_truth):
        """Evaluate the task. Stores the prompts, instances, outputs, and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
            instances: A dictionary of rows from the dataset.
            prompts: A list of prompts.
            ground_truth: A list of ground truths for the prompts.
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


class A1PHandlerCa(PHandlerCa):
    """Handler for the task of checking the total number of points that agent 1 recieved."""

    def generate_prompts(self, dataset_handler, model_handler):
        # get the instances from the dataset
        instances = dataset_handler.get_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model
        prompts = []
        for instance in instances:
            # isolate dialogue from data
            prompt = self.get_prompt_ca(instance, prompt_template, "mturk_agent_1")

            values = {item: level for level, item in instance['participant_info']['mturk_agent_1']['value2issue'].items()}

            fire_value = PRIORITY[values['Firewood']]
            food_value = PRIORITY[values['Food']]
            water_value = PRIORITY[values['Water']]

            prompt = prompt.replace("$fire_points$", str(fire_value))
            prompt = prompt.replace("$food_points$", str(food_value))
            prompt = prompt.replace("$water_points$", str(water_value))

            prompts.append(prompt)

        return(prompts, instances)

    def get_ground_truth(self, instances):
        """Get the ground truth for the task.

        Args:
            instances: A dictionary of rows from the dataset.
        """
        ground_truth = []
        for instance in instances:
            # get the points for agent 1
            points = instance['participant_info']['mturk_agent_1']['outcomes']['points_scored']
            ground_truth.append(str(points))

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
