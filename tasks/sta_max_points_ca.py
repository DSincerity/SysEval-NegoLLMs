"""
Task Question: What is the maximum possible points that you can get in any deal?

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: N
"""


from tasks.k_task import KBaseTaskHandler


class CaNDMaxPointsHandler(KBaseTaskHandler):
    """Handler for the CaSiNo No-Dialogue Max Points task of determining the maximum number of points each agent can possibly achieve."""

    possible_outputs = [str(num) for num in range(-40, 40)]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, utterance_bool=False, dialogue_bool=False, cot_bool=model_handler.args.use_cot)
        prompt_template = base_template.replace("$question$", "What is the maximum number of points that you can possibly get in any deal?").replace("$output_specification$", "Present your answer as a single number with no additional text.")

        return prompt_template

    def base_ground_truth(self, instances):
        """Get the priority dict for Agent 1.

        instances: the instances from the dataset.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        # return base ground truth: a list of "36"s because the maximum number of points an agent can receive is 3*5 + 3*4 + 3*3 = 36 in any instance.
        return ["36"] * len(instances)


"""
Test to see if the model can determine maximum points possible for Agent 1 in any deal using only value functions and no dialogue.
"""


class A1CaNDMaxPointsHandler(CaNDMaxPointsHandler):
    """Handler for the CaSiNo No-Dialogue Max Points task of determining maximum points possible for Agent 1."""

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        """Evaluate the task.

        Performs:
        1) Performance evaluation of the model on the dataset.
        2) Printing of aggregate results.
        3) Storing of the results and outputs.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        # get the instances and dialogues from the dataset
        instances = dataset_handler.get_instances()

        self.prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # self.prompt_template = prompt_template.replace("$agent$", "mturk_agent_1")

        prompts = []
        for instance in instances:
            prompt = self.get_prompt_ca(instance, self.prompt_template, "mturk_agent_1")
            # prompt = prompt.replace("mturk_agent_1", "Agent 1").replace("mturk_agent_2", "Agent 2")
            prompts.append(prompt)

        # get the ground truth for this task.
        ground_truth = self.base_ground_truth(instances)

        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        # get the model outputs - dict from prompt to the output. It's possible that some are missing so a dict is better than a list.
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
