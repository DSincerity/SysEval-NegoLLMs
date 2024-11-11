"""
Task Question: What is your highest priority issue?
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (partial dialogue history)
"""


from tasks.k_task import KBaseTaskHandler


class CaWCPrioritiesHandler(KBaseTaskHandler):
    """Handler for the CaSiNo With Counts Priorities task of determining how much the participants prefer food, water, and firewood."""

    possible_outputs = ["food", "water", "firewood"]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, utterance_bool=False, dialogue_bool=True, cot_bool=model_handler.cot)
        prompt_template = base_template.replace("$question$", "What is your highest priority issue?").replace("$output_specification$", "Present your answer as one of the following multiple choice options. You must select an option.\nA: food\nB: water\nC: firewood")

        return prompt_template

    def base_ground_truth(self, instances, agent):
        """Get the priority dict for the agent.

            agent: the agent we want to get the priority dict for.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        # return base ground truth: a list of dicts in the form {"Low": "Water", "Medium": "Food", "High": "Firewood"}.
        return [i["participant_info"][agent]["value2issue"] for i in instances]

    def a1_base_ground_truth(self, instances):
        """Get the priority dict for Agent 1.
        """

        return self.base_ground_truth(instances, "mturk_agent_1")

    def a2_base_ground_truth(self, instances):
            """Get the priority dict for Agent 2.
            """

            return self.base_ground_truth(instances, "mturk_agent_2")


class MidHigh1CaWCPrioritiesHandler(CaWCPrioritiesHandler):
    """Handler for the CaSiNo With Counts Priorities task of determining the lowest priority of Agent 1."""

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

        # get the instances from the dataset.
        instances = dataset_handler.get_instances()

        self.prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # self.prompt_template = prompt_template.replace("$agent$", "mturk_agent_1").replace("$priority_level$", "lowest")

        # make the respective prompts for each instance.
        prompts = []
        for instance in instances:
            prompt = self.get_prompt_ca(instance, self.prompt_template, "mturk_agent_1")
            # prompt = prompt.replace("mturk_agent_1", "Agent 1").replace("mturk_agent_2", "Agent 2")
            prompts.append(prompt)

        # get the ground truth for this task.
        base_ground_truth = self.a1_base_ground_truth(instances)
        ground_truth = [dict["High"].lower() for dict in base_ground_truth]

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
