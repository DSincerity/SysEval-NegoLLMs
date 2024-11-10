"""
Task Question: What quantity of books, hats, and balls does the speaker get in the proposal? (ask for a dictionary)

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (utterance, with some prior context)
"""


from tasks.k_task import KBaseTaskHandler


class DNDAllSlotsHandler(KBaseTaskHandler):
    """Handler for the DealOrNoDeal All Slots task of determining the entire deal proposed in an utterance."""

    possible_keys = ["books", "hats", "balls"]
    possible_outputs = [str(num) for num in range(-40, 40)]
    possible_outputs.append("NA")

    def get_prompt_template(self, dataset_handler, reg_or_con, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.
        
        Args:
            dataset_handler: the dataset handler.
            reg_or_con: describes whether the prompt should include context or not.
            model_handler: the model handler.
        """
        
        if reg_or_con == "reg":
            base_template = dataset_handler.get_utt_template(counts_bool=True, values_bool=True, context_bool=False, da_bool=False, cot_bool=model_handler.cot)
        elif reg_or_con == "con":
            base_template = dataset_handler.get_utt_template(counts_bool=True, values_bool=True, context_bool=True, da_bool=False, cot_bool=model_handler.cot)
            
        prompt_template = base_template.replace("$question$", "How many items does the speaker get for each issue in the proposal delimited by the <utterance> tags?").replace("$output_specification$", "Present your answer as a json within <answer> </answer> tags with keys as issues (books, hats, and balls) and values as the corresponding answers. If the answer is not clear for an issue, pick your best guess.")

        return prompt_template

    def extract_ground_truth(self, instances):
        """Get the slots and values from a certain proposal.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_propose_and_extra_utterances()
        # instances = [instance for instance in instances if instance["metadata"]["intent"] == "propose"]

        ground_truth = []

        for instance in instances:
            speaker = instance["agent"]
            proposal = instance["metadata"]["proposal"][str(speaker)]
            proposal = dict(sorted(proposal.items()))
            
            # replace with plural names
            new_proposal_dict = {
                "books": proposal["book"],
                "hats": proposal["hat"],
                "balls": proposal["ball"]
            }
            ground_truth.append(new_proposal_dict)

        # return ground truth: a list of string dicts in the form "{"books": 1, "hats": 0, "balls": 2}".
        return ground_truth


class DNDRegAllSlotsHandler(DNDAllSlotsHandler):
    """Handler for the DealOrNoDeal All Slots task of determining the entire deal proposed in an utterance without previous utterances as context."""

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
        instances = dataset_handler.get_propose_and_extra_utterances()
        instances = [instance for instance in instances if instance["metadata"]["intent"] == "propose"]

        prompt_template = self.get_prompt_template(dataset_handler, "reg", model_handler)

        self.prompt_template = prompt_template

        prompts = []
        for instance in instances:
            prompt = self.get_reg_slot_prompt_dnd(instance, self.prompt_template)
            
            # prompt = prompt.replace("YOU:", "Agent 1:").replace("THEM:", "Agent 2:").replace("Agent YOU", "Agent 1").replace("Agent THEM", "Agent 2")
            
            prompts.append(prompt)

        # get the ground truth for this task.
        ground_truth = self.extract_ground_truth(instances)

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