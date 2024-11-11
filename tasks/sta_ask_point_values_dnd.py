"""
Task Question: How many points is one item worth to you? (ask for a dict of issue:point_value)

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: N
"""


from tasks.k_task import KBaseTaskHandler


class DNDNDPointValuesHandler(KBaseTaskHandler):
    """Handler for the DealOrNoDeal No-Dialogue Point Values task of determining how many points each item (i.e. a book, a ball, and a hat) is worth to each of the participants when given only value functions and no dialogue."""

    possible_keys = ["books", "hats", "balls"]
    possible_outputs = [str(num) for num in range(-40, 40)]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, dialogue_bool=False, da_bool=False, cot_bool=model_handler.cot)
        prompt_template = base_template.replace("$question$", "How many points is one item of each issue worth to you?").replace("$output_specification$", "Present your answer as a json within <answer> </answer> tags with keys as issues (books, hats, and balls) and values as the corresponding answers.")

        return prompt_template

    # merged with below fn.
    # def base_ground_truth(self, input_type, dataset_handler):
    #     """Get the [book-points, hat-points, ball-points] list for an agent.

    #     Args:
    #         input_type: describes the agent whose value function is being determined.
    #         dataset_handler: the dataset handler.
    #     """

    #     # get the instances from the dataset.
    #     instances = dataset_handler.get_instances()

    #     # return the base ground truth: a list of lists in the form [book-points, hat-points, ball-points].
    #     return [i[input_type]["value"] for i in instances]

    def you_base_ground_truth(self, input_type, instances):
        """Get the [book-points, hat-points, ball-points] list for Agent YOU.

        Args:
            dataset_handler: the dataset handler.
        """
        assert input_type == "input"
        ground_truth = []
        for instance in instances:
            gt = {
                "books": instance[input_type]["value"][0],
                "hats": instance[input_type]["value"][1],
                "balls": instance[input_type]["value"][2]
            }
            ground_truth.append(gt)

        return ground_truth

    # def them_base_ground_truth(self, dataset_handler):
    #     """Get the [book-points, hat-points, ball-points] list for Agent THEM.

    #     Args:
    #         dataset_handler: the dataset handler.
    #     """

    #     return self.base_ground_truth("partner_input", dataset_handler)


class BYDNDNDPointValuesHandler(DNDNDPointValuesHandler):
    """Handler for the DealOrNoDeal No-Dialogue Point Values task of determining how many points a book is worth to Agent YOU."""

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

        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # not required anymore
        # self.prompt_template = prompt_template.replace("$agent_name$", "YOU").replace("$item_type$", "book")

        prompts = []
        for instance in instances:
            prompt = self.get_prompt_dnd(instance, prompt_template, "YOU")

            # skip this - not required
            # prompt = prompt.replace("YOU:", "Agent 1:").replace("THEM:", "Agent 2:").replace("Agent YOU", "Agent 1").replace("Agent THEM", "Agent 2")

            prompts.append(prompt)

        # get the ground truth for this task.
        ground_truth = self.you_base_ground_truth("input", instances)

        # ground_truth = [str(lst[0]) for lst in base_ground_truth]

        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        # get the model outputs - dict from prompt to the output. It's possible that some are missing so a dict is better than a list.
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
