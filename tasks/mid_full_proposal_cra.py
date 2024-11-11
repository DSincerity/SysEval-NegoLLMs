"""
Task Question: What quantity of books, hats, and balls does the speaker get in the proposal? (ask for a dictionary)

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (utterance, with some prior context)
"""


from tasks.k_task import KBaseTaskHandler


class CRAAllSlotsHandler(KBaseTaskHandler):
    """Handler for the CRA All Slots task of determining the entire deal proposed in an utterance."""

    possible_keys = ["painting", "lamp", "record"]
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
            base_template = dataset_handler.get_utt_template(context_bool=False, da_bool=False, cot_bool=model_handler.cot)
        elif reg_or_con == "con":
            base_template = dataset_handler.get_utt_template(context_bool=True, da_bool=False, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "How many items does the speaker get for each issue in the proposal delimited by the <utterance> tags?").replace("$output_specification$", "Present your answer as a json within <answer> </answer> tags with keys as issues (painting, lamp, and record) and values as the corresponding answers. If the answer is not clear for an issue, output \"NA\".")

        return prompt_template

    def ground_truth(self, dataset_handler):
        """Get the ground truth for instances that involve annotated proposals (i.e. output by self.get_slot_instances()).

        Args:
            dataset_handler: the dataset handler.
        """

        instances = dataset_handler.get_slot_instances()

        ground_truth = []
        for instance in instances:
            deal_str = instance["DUD"]

            deal_dict = {}
            if "P" in deal_str:
                painting_index = deal_str.index("P")

                if instance["spkr"] == "A":
                    deal_dict["painting"] = int(deal_str[painting_index + 1])
                elif instance["spkr"] == "B":
                    deal_dict["painting"] = int(deal_str[painting_index + 3])
            else:
                deal_dict["painting"] = "NA"

            if "L" in deal_str:
                lamp_index = deal_str.index("L")

                if instance["spkr"] == "A":
                    deal_dict["lamp"] = int(deal_str[lamp_index + 1])
                elif instance["spkr"] == "B":
                    deal_dict["lamp"] = int(deal_str[lamp_index + 3])
            else:
                deal_dict["lamp"] = "NA"

            if "R" in deal_str:
                record_index = deal_str.index("R")

                if instance["spkr"] == "A":
                    deal_dict["record"] = int(deal_str[record_index + 1])
                elif instance["spkr"] == "B":
                    deal_dict["record"] = int(deal_str[record_index + 3])
            else:
                deal_dict["record"] = "NA"

            ground_truth.append(deal_dict)

        return ground_truth


class CRARegAllSlotsHandler(CRAAllSlotsHandler):
    """Handler for the CRA All Slots task of determining the entire deal proposed in an utterance without previous utterances as context."""

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

        # get the instances from the dataset
        instances = dataset_handler.get_slot_instances()

        prompt_template = self.get_prompt_template(dataset_handler, "reg", model_handler)

        self.prompt_template = prompt_template

        # get the ground truth for this task.
        ground_truth = self.ground_truth(dataset_handler)

        print(len(instances), len(ground_truth))
        assert len(instances) == len(ground_truth)

        instances2, ground_truth2 = [], []
        for inst, gt in zip(instances, ground_truth):
            if isinstance(inst["utt"], str):
                instances2.append(inst)
                ground_truth2.append(gt)

        instances = instances2
        ground_truth = ground_truth2

        print(len(instances), len(ground_truth))
        assert len(instances) == len(ground_truth)

        prompts = []
        for instance in instances:
            prompt = self.get_reg_ul_prompt_cra(instance["utt"], self.prompt_template)
            prompt = prompt.replace("Alice", "YOU").replace("Bob", "THEM")
            prompts.append(prompt)

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
