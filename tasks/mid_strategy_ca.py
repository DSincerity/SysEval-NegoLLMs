"""
Task Question: Which negotiation strategies are employed in the utterance? (returns a list of strategies)

Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (utterance, with some prior context)
"""


import copy
from tasks.task_w import WBaseTaskHandler


class NSUHandler(WBaseTaskHandler):
    """Handler for the task of identifying the negotiation strategies present in
    each utterance."""

    possible_outputs =  ["small-talk", "empathy", "coordination", "no-need", "elicit-pref", "uv-part", "vouch-fair", "self-need", "other-need"]

    def get_prompt_template(self, dataset_handler, model_handler):

        base_template = dataset_handler.get_utt_template(counts_bool=True, values_bool=True, utterance_bool=True, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "Which negotiation strategies are employed in the utterance?").replace("$output_specification$", "Present your answer as a comma-separated list of strategies, contained in <answer> </answer> tags with no additional text.")

        return prompt_template


    def find_spk_str(self, instance, curr_txt):
        """Go through the chat and find the speaker string for this utterance."""

        for item in instance['chat_logs']:
            if item['text'] == curr_txt:
                if item['id'] == "mturk_agent_1":
                    return "YOU: "
                else:
                    return "THEM: "

        raise ValueError


    def get_prev_utterances(self, instance, curr_txt):
        """Find sequence of prev utterances with speakers."""

        prev_utts = []
        for item in instance['annotations']:
            if item[0] != curr_txt:
                prev_utts.append(item[0])
            else:
                break

        if len(prev_utts) > self.args.num_prior_utts:
            prev_utts = prev_utts[-self.args.num_prior_utts:]

        # attach speaker str for all prev_utts
        prev_utts = [self.find_spk_str(instance, utt) + utt for utt in prev_utts]

        return " ".join(prev_utts)


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
            annotations = instance['annotations']
            for index in range(len(annotations)):

                if "non-strategic" in annotations[index][-1]:
                    # skip non-strategic utterances
                    continue

                if self.args.num_prior_utts == 0:
                    prompt = prompt_template.replace("$utterance$", annotations[index][0])
                else:
                    spk_str = self.find_spk_str(instance, annotations[index][0])
                    prompt = prompt_template.replace("$utterance$", spk_str + annotations[index][0])
                    prompt = prompt.replace("$previous_utterance$", self.get_prev_utterances(instance, annotations[index][0]))

                # fill in the values.
                agent1_dict = instance['participant_info']["mturk_agent_1"]["value2issue"]
                agent1_switched = {item: level for level, item in agent1_dict.items()}

                agent1_points = copy.deepcopy(agent1_switched)
                for k, v in agent1_switched.items():
                    if v == 'Low':
                        agent1_points[k] = 3
                    elif v == 'Medium':
                        agent1_points[k] = 4
                    else:
                        agent1_points[k] = 5
                prompt = prompt.replace("$food_points$", str(agent1_points['Food']))
                prompt = prompt.replace("$water_points$", str(agent1_points['Water']))
                prompt = prompt.replace("$fire_points$", str(agent1_points['Firewood']))

                prompts.append(prompt)

                if "," in annotations[index][-1]:
                    gt_str = annotations[index][-1].replace("promote-coordination", "coordination")

                    strat = gt_str.split(",")
                    ground_truth.append(strat)
                else:
                    ground_truth.append([annotations[index][-1].replace("promote-coordination", "coordination")])

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

            answer = outputs_dict[prompt].replace("undervalue-partner", "uv-part").replace("vouch-fairness", "vouch-fair")

            p = []
            for option in self.possible_outputs:
                if option in answer:
                    p.append(option)
            final_predictions.append(p)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances
