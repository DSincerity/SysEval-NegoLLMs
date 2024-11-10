"""
Task Question: What is your lowest priority issue?

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: N
"""


from tasks.k_task import KBaseTaskHandler


class JIWCPrioritiesHandler(KBaseTaskHandler):
    """Handler for the JobInterview With-Counts Priorities task of determining the priorities of the negotiators."""

    possible_outputs = ["position", "company", "salary", "days_off", "workplace"]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.
        
        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        base_template = dataset_handler.get_dial_template(counts_bool=True, cot_bool=model_handler.cot, values_bool=True, dialogue_bool=False, full_dialogue_bool=False)

        prompt_template = base_template.replace("$question$", "What is your lowest priority issue?").replace("$output_specification$", "Present your answer as one of the following multiple choice options. You must select an option.\nA: position\nB: company\nC: salary\nD: days_off\nE: workplace")

        return prompt_template

    def base_ground_truth(self, agent, instances):
        """Determine the agent's weights for each issue.

        Args:
            agent: the agent whose weights are being determined.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        base_ground_truth = []
        for instance in instances:
            if instance.users[0].context["role"] == agent:
                weights_list = {dict["name"]: dict["weight"] for dict in instance.users[0].context["utilities"]}
                base_ground_truth.append(weights_list)
            elif instance.users[1].context["role"] == agent:
                weights_list = {dict["name"]: dict["weight"] for dict in instance.users[1].context["utilities"]}
                base_ground_truth.append(weights_list)

        # Position and Company are combined because this is how their are shown in the UI of the MTurk task.
        for dict in base_ground_truth:
            dict["position"] = dict["Position"]
            del dict["Position"]

            dict["company"] = dict["Company"]
            del dict["Company"]

            dict["salary"] = dict["Salary"]
            del dict["Salary"]

            dict["days_off"] = dict["Weekly holiday"]
            del dict["Weekly holiday"]

            dict["workplace"] = dict["Workplace"]
            del dict["Workplace"]
        
        return base_ground_truth
    
    def w_base_ground_truth(self, instances):
        """Determine the worker's weights for each issue.
        """

        return self.base_ground_truth("worker", instances)
    
    def r_base_ground_truth(self, instances):
        """Determine the recruiter's weights for each issue.
        """

        return self.base_ground_truth("recruiter", instances)


class WLowJIPrioritiesHandler(JIWCPrioritiesHandler):
    """Handler for the JobInterview with-Counts Priorities task of determining the highest priority issue of the worker."""

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

        # get the instances from the dataset.
        instances = dataset_handler.get_instances()

        self.prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # self.prompt_template = prompt_template.replace("$worker/recruiter$", "worker").replace("$priority_level$", "highest")

        # make the respective prompts for each instance.
        prompts = []
        for instance in instances:
            prompt = self.get_prompt_with_bids_ji(instance, self.prompt_template)
            prompts.append(prompt)

        # get the ground truth for this task.
        base_ground_truth = self.w_base_ground_truth(instances)

        lowest_ground_truth = []
        for dict in base_ground_truth:
            h_chosen_key = [key for key, value in dict.items() if value == min(dict.values())][0]
            lowest_ground_truth.append(h_chosen_key)

        ground_truth = lowest_ground_truth

        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        # get the model outputs - dict from prompt to the output. It's possible that some are missing so a dict is better than a list.
        outputs_dict = model_handler.get_model_outputs(new_prompts, new_ground_truth)
        
        #only for the ones that are unique and where valid predictions are available
        final_prompts, final_predictions, final_ground_truth = self.get_final_outputs(outputs_dict, self.possible_outputs, new_prompts, new_ground_truth)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        # store the results and outputs in a json file
        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)
        
        return instances