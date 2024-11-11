"""
Task Question: Which dialogue acts are employed in the utterance? (returns a list of dialogue acts)
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (utterance, with some prior context)
"""


from tasks.task_w import WBaseTaskHandler


class DASUHandler(WBaseTaskHandler):
    """Handler for the task of annotating an utterance with the correct dialogue act."""

    possible_outputs =  ["greet", "inquire", "propose", "agree", "disagree", "insist", "unknown"]

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_utt_template(counts_bool=True, values_bool=True, context_bool=False, da_bool=True, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "Which dialogue act is employed in the utterance contained in <utterance> tags?").replace("$output_specification$", "Present your answer as a single word.")

        return prompt_template

    def get_prev_utterances(self, instance, turn_ix):
        """
        Get the previous utterances in the dialogue.
        """
        prev_utterances = []
        for prev_turn in instance['events'][:turn_ix]:
            if type(prev_turn['data']) != dict:
                spk = "YOU: "
                if prev_turn["agent"] == 1:
                    spk = "THEM: "
                prev_utterances.append(spk + prev_turn['data'])

        if len(prev_utterances) > self.args.num_prior_utts:
            prev_utterances = prev_utterances[-self.args.num_prior_utts:]

        return " ".join(prev_utterances)

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        """Evaluate the task. Stores the prompts, instances, outputs,
        and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
        """

        # get the instances from the dataset
        instances = dataset_handler.get_da_instances()
        prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # create a list of prompts for the model and hard code ground truth
        prompts = []
        ground_truth = []
        for instance in instances:
            for turn_ix, turn in enumerate(instance['events']):
                if type(turn['data']) != dict:

                    if self.args.num_prior_utts == 0:
                        prompt = prompt_template.replace("$utterance$", turn['data'])
                    else:
                        spk = "YOU: "
                        if turn["agent"] == 1:
                            spk = "THEM: "
                        prompt = prompt_template.replace("$utterance$", spk + turn['data'])
                        prompt = prompt.replace("$previous_utterance$", self.get_prev_utterances(instance, turn_ix))

                    assert "$num_books$" in prompt
                    prompt = prompt.replace("$num_books$", str(instance["scenario"]["kbs"][0][0]["Count"]))
                    prompt = prompt.replace("$num_hats$", str(instance["scenario"]["kbs"][0][1]["Count"]))
                    prompt = prompt.replace("$num_balls$", str(instance["scenario"]["kbs"][0][2]["Count"]))

                    assert "$book_points$" in prompt
                    prompt = prompt.replace("$book_points$", str(instance["scenario"]["kbs"][0][0]["Value"]))
                    prompt = prompt.replace("$hat_points$", str(instance["scenario"]["kbs"][0][1]["Value"]))
                    prompt = prompt.replace("$ball_points$", str(instance["scenario"]["kbs"][0][2]["Value"]))

                    prompts.append(prompt)
                    ground_truth.append(turn['metadata']['intent'])

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
