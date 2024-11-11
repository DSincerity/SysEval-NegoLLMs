"""
Task Question: Given the partial dialogue, generate your next response in the negotiation.

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (partial dialogue history)
"""


from tasks.task_w import WBaseTaskHandler


class GSCaHandler(WBaseTaskHandler):

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, utterance_bool=False, dialogue_bool=True, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "Given the recent dialogue history inside <dialogue> tags, generate your next response in the negotiation concisely, following a similar style as previous utterances.").replace("$output_specification$", "")

        return prompt_template

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        """Evaluate the task. Stores the prompts, instances, outputs,
        and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
        """

        # get the instances from the dataset
        instances = dataset_handler.get_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model and hard code ground truth
        prompts = []
        ground_truth = []
        for instance in instances:
            len_chat = len(instance['chat_logs'])

            for i in range(len_chat):
                if instance["chat_logs"][i]["id"] != "mturk_agent_1":
                    # skip the other agent's turn
                    continue

                if instance["chat_logs"][i]["text"] in ["Submit-Deal", "Reject-Deal", "Accept-Deal", "Walk-Away"]:
                    # skip the special utterances
                    continue

                actual_resp = instance['chat_logs'][i]['text']

                ground_truth.append(actual_resp)
                prompt = self.get_partial_dial_ca(i, instance, prompt_template)

                assert "mturk_agent_" not in prompt

                prompts.append(prompt)

        # get the model outputs - dict from prompt to the output.
        # It's possible that some are missing so a dict is better than a list.
        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        outputs_dict = model_handler.get_model_outputs(new_prompts, new_ground_truth)

        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(new_prompts, new_ground_truth):
            if prompt not in outputs_dict:
                continue

            final_prompts.append(prompt)
            final_ground_truth.append(gt)

            final_predictions.append(outputs_dict[prompt])

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances
