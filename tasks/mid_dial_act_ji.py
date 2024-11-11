"""
Task Question: Which dialogue acts are employed in the utterance? (returns a list of dialogue acts)
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y (utterance, with some prior context)
"""


from tasks.k_task import KBaseTaskHandler
import copy

class JIDAHandler(KBaseTaskHandler):
    """Handler for the JobInterview Dialogue Acts task of determining the coarse dialogue act of an utterance."""

    labels = ["greet", "inquire", "propose", "disagree", "agree", "inform", "unknown"]

    def get_prompt_template(self, dataset_handler, reg_or_con, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            reg_or_con: describes whether the prompt should include context or not.
            model_handler: the model handler.
        """

        if reg_or_con == "reg":
            base_template = dataset_handler.get_utt_template(context_bool=False, full_dial_bool=False, cot_bool=model_handler.cot, counts_bool=True, values_bool=True, da_bool=True)
        elif reg_or_con == "con":
            base_template = dataset_handler.get_utt_template(context_bool=True, full_dial_bool=False, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "Which dialogue acts are employed in the utterance delimited by the <utterance> tags?").replace("$output_specification$", "Present your answer as a Python list of the relevant options. At least one option applies.")

        return prompt_template

    def get_full_dial_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """
        base_template = dataset_handler.get_utt_template(context_bool=False, full_dial_bool=True, cot_bool=model_handler.cot)

        prompt_template = base_template.replace("$question$", "Which dialogue act(s) listed in the <options> tags are employed in EACH UTTERANCE of the dialogue delimited by the <dialogue> tags?").replace("$output_specification$", "Present your answer as a Python list of tuples, where the first value of each tuple is an utterance such as 'worker: I want a higher salary.' and the second value of each tuple is a list of the dialogue acts in that utterance. At least one dialogue act applies to each utterance. Your list should contain $num_utterances$ tuples.")

        return prompt_template

    def get_da_dialogue_ground_truth(self, dataset_handler):
        """Get the ground truth for dialogue-act tasks (i.e. dialogue-act annotations) in which a whole dialogue is input.

        Args:
            dataset_handler: the dataset handler.
        """

        # for the first 10 instances from the dataset; each list of tags corresponds to a full dialogue.
        instances_da = dataset_handler.da_list[:self.args.num_instances]

        dialogue_annotations = []
        for dict in instances_da:
            dialogue_anns_str = dict["meta_text"]
            dialogue_anns_with_seps = dialogue_anns_str.split(",")

            for tag_str in dialogue_anns_with_seps:
                dialogue_annotations.append(tag_str)

        ground_truth = []
        organized_dialogue_tags = []
        turn_tags = []

        for tag in dialogue_annotations:
            if "<end>" in tag:
                if turn_tags:
                    organized_dialogue_tags.append(turn_tags)
                    turn_tags = []

                ground_truth.append(organized_dialogue_tags)
                organized_dialogue_tags = []
            if "<sep>" in tag:
                if turn_tags:
                    organized_dialogue_tags.append(turn_tags)
                    turn_tags = []
            elif "<greet>" in tag:
                turn_tags.append("greet")
            elif "<inquire>" in tag:
                turn_tags.append("inquire")
            elif "<propose>" in tag:
                turn_tags.append("propose")
            elif "<agree>" in tag:
                turn_tags.append("agree")
            elif "<disagree>" in tag:
                turn_tags.append("disagree")
            elif "<inform>" in tag:
                turn_tags.append("inform")
            elif "<unknown>" in tag:
                turn_tags.append("unknown")

        # ground_truth is a list. Within this, each dialogue has its own list. A dialogue's list contains a list of tags for each turn in the dialogue.
        return ground_truth

    def get_da_turn_ground_truth(self, dataset_handler):
        """Get the ground truth for dialogue-act tasks (i.e. dialogue-act annotations) in which a single turn is input.

        Args:
            dataset_handler: the dataset handler.
        """

        base_ground_truth = self.get_da_dialogue_ground_truth(dataset_handler)

        ground_truth = []
        for organized_dialogue_tags in base_ground_truth:
            for turn_tags in organized_dialogue_tags:
                ground_truth.append(turn_tags)

        # ground truth is a list of lists. Each list within ground truth has the tags corresponding to a single turn.
        return ground_truth


class JIRegDAHandler(JIDAHandler):
    """Handler for the JobInterview Dialogue Acts task of determining the coarse dialogue act of an utterance without previous utterances as context."""


    def get_prev_utterances(self, turns, turn):
        """Get prev utterances for this turn."""

        def get_turn_str(tt):
            """Convert the turn list of phrases to a single string."""

            turn_string = ""
            list_of_phrases = tt

            if "worker: " in list_of_phrases[0]:
                speaker = "worker: "
            elif "recruiter: " in tt[0]:
                speaker = "recruiter: "

            for phrase in list_of_phrases:
                turn_string += phrase.replace("worker: ", "").replace("recruiter: ", "") + "\n"

            turn_string = speaker + turn_string
            return turn_string

        prev_utterances = []

        curr_turn_str = get_turn_str(turn)
        for prev_turn in turns:

            prev_turn_str = get_turn_str(prev_turn)

            if prev_turn_str != curr_turn_str:
                prev_utterances.append(prev_turn_str)
            else:
                break

        if len(prev_utterances) > self.args.num_prior_utts:
            prev_utterances = prev_utterances[-self.args.num_prior_utts:]

        return " ".join(prev_utterances)


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

        # get the instances (i.e. turns) from the dataset
        # turns = self.get_turns(dataset_handler)
        d_wise_ground_truth = self.get_da_dialogue_ground_truth(dataset_handler)
        all_instances = dataset_handler.get_instances()[:self.args.num_instances]
        assert len(d_wise_ground_truth) == len(all_instances)

        flat_turns, flat_ground_truth, flat_prev_turns = [], [], []
        for instance, dwgt in zip(all_instances, d_wise_ground_truth):
            turns = self.get_turns_from_dialogue_ji(instance)
            if len(turns) != len(dwgt):
                print("Mismatched lengths: ", len(turns), len(dwgt))
                continue

            for turn, gt in zip(turns, dwgt):

                turn_obj = {
                    "turn_comments": turn,
                    "orig_instance": copy.deepcopy(instance),
                }
                flat_turns.append(turn_obj)
                flat_ground_truth.append(gt)
                flat_prev_turns.append(self.get_prev_utterances(turns, turn))

        print("Flat lens: ", len(flat_turns), len(flat_ground_truth), len(flat_prev_turns))
        assert len(flat_turns) == len(flat_ground_truth) == len(flat_prev_turns)

        self.prompt_template = self.get_prompt_template(dataset_handler, "reg", model_handler)

        prompts = []
        for turn, prev_turn_str in zip(flat_turns, flat_prev_turns):
            prompt = self.get_reg_da_prompt_ji(turn, self.prompt_template)
            if self.args.num_prior_utts > 0:
                prompt = prompt.replace("$previous_utterance$", prev_turn_str)
            prompts.append(prompt)

        assert len(prompts) == len(flat_ground_truth)

        ground_truth = flat_ground_truth[:]

        # remove cases where the prompt contains structured utterances
        prompts2, ground_truth2 = [], []
        for pt, turn_obj, gt in zip(prompts, flat_turns, ground_truth):
            if "< reject bid >" not in turn_obj["turn_comments"][0] and "< accept bid >" not in turn_obj["turn_comments"][0] and "< propose >" not in turn_obj["turn_comments"][0]:
                prompts2.append(pt)
                ground_truth2.append(gt)
        prompts, ground_truth = prompts2, ground_truth2

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

        return turns
