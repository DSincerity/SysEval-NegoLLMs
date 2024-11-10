"""
Task Question: How many points is one item worth to you? (ask for a dict of issue:point_value)

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: N
"""


from tasks.k_task import KBaseTaskHandler


class CaNDPointValuesHandler(KBaseTaskHandler):
    """Handler for the CaSiNo No-Dialogue Point Values task of determining how many points each item (i.e. a food package, a water package, and a firewood package) is worth to each of the participants when given only value functions and no dialogue."""

    possible_keys = ["food", "water", "firewood"]
    possible_outputs = [str(num) for num in range(-40, 40)]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.
        
        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, utterance_bool=False, dialogue_bool=False, cot_bool=model_handler.cot)
        prompt_template = base_template.replace("$question$", "How many points is one package of each issue worth to you?").replace("$output_specification$", "Present your answer as a json within <answer> </answer> tags with keys as issues (food, water, and firewood) and values as the corresponding answers.")

        return prompt_template

    def base_ground_truth(self, instances, agent):
        """Get the item-and-points dict for the agent.

        Args:
            agent: the agent we want to get the item-and-points dict for.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        # a list of priority dicts in the form {'Low': 'Water', 'Medium': 'Food', 'High': 'Firewood'}.
        true_priorities = [i["participant_info"][agent]["value2issue"] for i in instances]
        
        # get a list of dicts in the form {'Water': 'Low', 'Food': 'Medium', 'Firewood': 'High'}.
        items_first = []
        for priority_dict in true_priorities:
            switched = {item: level for level, item in priority_dict.items()}
            items_first.append(switched)

        # convert the priority levels to point values to get a list of dicts in the form {'Water': 3, 'Food': 4, 'Firewood': 5}.
        def priority2points(dict):
            dict2 = {}
            for k, v in dict.items():
                if v == 'Low':
                    dict2[k.lower()] = "3"
                elif v == 'Medium':
                    dict2[k.lower()] = "4"
                else:
                    dict2[k.lower()] = "5"
            return dict2
        
        # return base ground truth: a list of dicts in the form {"Water": 3, "Food": 4, "Firewood": 5}.
        return [priority2points(dict) for dict in items_first]

    def a1_base_ground_truth(self, instances):
        """Get the item-and-points dict for Agent 1.
        """

        return self.base_ground_truth(instances, "mturk_agent_1")
    
    def a2_base_ground_truth(self, instances):
        """Get the item-and-points dict for Agent 2.

        """

        # get the instances from the dataset.
        
        return self.base_ground_truth(instances, "mturk_agent_2")


class Food1CaNDPointValuesHandler(CaNDPointValuesHandler):
    """Handler for the CaSiNo No-Dialogue Point Values task of determining how many points a food package is worth to Agent 1."""

    def evaluate(self, dataset_handler, model_handler):
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

        # self.prompt_template = prompt_template.replace("$agent$", "mturk_agent_1").replace("$item_type$", "food")

        prompts = []
        for instance in instances:
            prompt = self.get_prompt_ca(instance, self.prompt_template, "mturk_agent_1")
            # prompt = prompt.replace("mturk_agent_1", "Agent 1").replace("mturk_agent_2", "Agent 2")
            prompts.append(prompt)

        # get the ground truth for this task.
        ground_truth = self.a1_base_ground_truth(instances)
        # ground_truth = [dict["Food"] for dict in base_ground_truth]

        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

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