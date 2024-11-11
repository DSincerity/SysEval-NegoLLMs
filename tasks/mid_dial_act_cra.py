"""
Task Question: Which dialogue acts are employed in the utterance? (returns a list of dialogue acts)
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (utterance, with some prior context)
"""


from tasks.k_task import KBaseTaskHandler


class CRADAHandler(KBaseTaskHandler):
    """Handler for the CRA Dialogue Acts task of determining the coarse dialogue act of an utterance."""

    labels = ["make offer", "ask offer", "accept", "reject", "ask preference", "share preference"]

    def get_prompt_template(self, dataset_handler, reg_or_con, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            reg_or_con: describes whether the prompt should include context or not.
            model_handler: the model handler.
        """

        if reg_or_con == "reg":
            base_template = dataset_handler.get_utt_template(context_bool=False, da_bool=True, cot_bool=model_handler.cot)
        elif reg_or_con == "con":
            base_template = dataset_handler.get_utt_template(context_bool=True, da_bool=True, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "Which dialogue acts are employed in the utterance delimited by the <utterance> tags?").replace("$output_specification$", "Present your answer as a Python list of the relevant options. At least one option applies.")

        return prompt_template


class CRARegDAHandler(CRADAHandler):
    """Handler for the CRA Dialogue Acts task of determining the coarse dialogue act of an utterance without previous utterances as context."""

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
        instances = dataset_handler.get_da_instances()

        prompt_template = self.get_prompt_template(dataset_handler, "reg", model_handler)

        self.prompt_template = prompt_template

        # get the ground truth for this task.
        ground_truth = dataset_handler.get_da_ground_truth()

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
            if self.args.num_prior_utts == 0:
                prompt = self.get_reg_ul_prompt_cra(instance["utt"], self.prompt_template)
                prompt = prompt.replace("Alice", "YOU").replace("Bob", "THEM")
            else:
                spk = instance["spkr"]
                if spk == "A":
                    spk = "Alice: "
                elif spk == "B":
                    spk = "Bob: "
                else:
                    raise Exception("Invalid speaker")
                uttt = spk + instance["utt"]
                cxtt = instance["context_str"]
                prompt = self.get_con_ul_prompt_cra(uttt, cxtt, self.prompt_template)
                prompt = prompt.replace("Alice", "YOU").replace("Bob", "THEM")
            prompts.append(prompt)

        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        # get the model outputs - dict from prompt to the output. It's possible that some are missing so a dict is better than a list.
        outputs_dict = model_handler.get_model_outputs(new_prompts, new_ground_truth)

        #only for the ones that are unique and where valid predictions are available
        final_prompts, final_predictions, final_ground_truth = self.get_final_outputs_ann(outputs_dict, self.labels, new_prompts, new_ground_truth)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances
