"""
Base class for all tasks or tests.
"""

import copy
import json
import utils
import ast


class KBaseTaskHandler:
    """Base handler for every task."""

    def __init__(self, name, args):
        """Initialize the task handler."""
        self.name = name
        self.args = args

    def evaluate(self, dataset_handler, model_handler):
        """Primary method to evaluate a task on a given model and dataset.
        Stores instances, prompts, outputs, and ground truth.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """
        raise NotImplementedError

    def flatten(self, lst):
        """Flatten a list by one level (e.g. turn a list of list of lists into a list of lists).

        Args:
            lst: a list."""

        new = []

        for item in lst:
            for element in item:
                new.append(element)

        return new

    def get_prompt_ca(self, instance, template, agent):
        """Method to get a prompt given a single instance from the CaSiNo
        dataset.

        Args:
            instance: a list of dicts containing chat_logs.
            template: the prompt template for a particular task.
            agent: the agent the prompt is asking about.
        """

        dialogue = ""
        logs = instance['chat_logs']
        participant_info = instance['participant_info']

        history = logs[:]

        # remove special utterances
        history2 = []
        for utt in history:
            if utt['text'] not in ["Submit-Deal", "Accept-Deal", "Walk-Away", "Reject-Deal"]:
                history2.append(utt)

        # check if we need to only use a partial dialogue
        if self.args.num_utts_partial_dial != -1:
            # use first k utterances only, from the start of the dialogue
            history2 = history2[:self.args.num_utts_partial_dial]

        for utt in history2:
            dialogue += utt['id'] + ": " + utt['text'] + "\n"

        # use you and them - same as DND. tread mturk_agent_1 as YOU
        dialogue = dialogue.replace("mturk_agent_1:", "YOU:")
        dialogue = dialogue.replace("mturk_agent_2:", "THEM:")

        # a priority dict in the form {'Low': 'Water', 'Medium': 'Food', 'High': 'Firewood'}.
        agent1_dict = participant_info["mturk_agent_1"]["value2issue"]
        agent2_dict = participant_info["mturk_agent_2"]["value2issue"]

        # get dicts in the form {'Water': 'Low', 'Food': 'Medium', 'Firewood': 'High'}.
        agent1_switched = {item: level for level, item in agent1_dict.items()}
        agent2_switched = {item: level for level, item in agent2_dict.items()}

        # convert the priority levels to point values to get dicts in the form {'Water': 3, 'Food': 4, 'Firewood': 5}.
        def priority2points(dict):
            for k, v in dict.items():
                if v == 'Low':
                    dict[k] = 3
                elif v == 'Medium':
                    dict[k] = 4
                else:
                    dict[k] = 5
            return dict

        agent1_points = priority2points(agent1_switched)
        agent2_points = priority2points(agent2_switched)

        prompt = template.replace("$dialogue$", dialogue)
        if agent == "mturk_agent_1":
            prompt = prompt.replace("$food_points$", str(agent1_points['Food']))
            prompt = prompt.replace("$water_points$", str(agent1_points['Water']))
            prompt = prompt.replace("$fire_points$", str(agent1_points['Firewood']))
        elif agent == "mturk_agent_2":
            prompt = prompt.replace("$food_points$", str(agent2_points['Food']))
            prompt = prompt.replace("$water_points$", str(agent2_points['Water']))
            prompt = prompt.replace("$fire_points$", str(agent2_points['Firewood']))

        return prompt

    def get_prompt_dnd(self, instance, template, agent):
        """Method to get a prompt given a template and instance from the DealOrNoDeal
        dataset.

        Args:
            instance: a dict containing a row from the dataset.
            template: the prompt template for a particular task.
            agent: the agent the prompt is asking about.
        """

        dialogue = ""
        dialogue_list = str(instance['dialogue']).split(" <eos> ")

        # skip the last selection utterance
        for turn in dialogue_list[:-1]:
            dialogue += turn + "\n"

        you_value = instance['input']["value"]
        them_value = instance['partner_input']["value"]
        counts = instance['input']['count']

        prompt = template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$num_books$", str(counts[0]))
        prompt = prompt.replace("$num_hats$", str(counts[1]))
        prompt = prompt.replace("$num_balls$", str(counts[2]))

        if agent == "YOU":
            prompt = prompt.replace("$book_points$", str(you_value[0]))
            prompt = prompt.replace("$hat_points$", str(you_value[1]))
            prompt = prompt.replace("$ball_points$", str(you_value[2]))
        elif agent == "THEM":
            prompt = prompt.replace("$book_points$", str(them_value[0]))
            prompt = prompt.replace("$hat_points$", str(them_value[1]))
            prompt = prompt.replace("$ball_points$", str(them_value[2]))

        return prompt

    def get_ul_prompt_dnd(self, instance, template):
        """Method to get a prompt given a template and instance from the DealOrNoDeal
        dataset for utterance-level tasks.

        Args:
            instance: a dict containing a row from the dataset.
            template: the prompt template for a particular task.
        """
        if "$dialogue$" in template:
            dialogue = ""
            for utterance_dict in instance:
                if utterance_dict["agent"] == 0:
                    dialogue += "YOU: " + utterance_dict["data"] + "\n"
                elif utterance_dict["agent"] == 1:
                    dialogue += "THEM: " + utterance_dict["data"] + "\n"
            prompt = template.replace("$dialogue$", dialogue)
        elif "$utterance$" in template:
            if instance["agent"] == 0:
                prompt = template.replace("$utterance$", "YOU: " + instance["data"].strip())
            elif instance["agent"] == 1:
                prompt = template.replace("$utterance$", "THEM: " + instance["data"].strip())
        return prompt

    def get_reg_slot_prompt_dnd(self, instance, template):
        base_prompt = self.get_ul_prompt_dnd(instance, template)
        book_count, hat_count, ball_count = instance["Counts"]["book"], instance["Counts"]["hat"], instance["Counts"]["ball"]
        prompt = base_prompt.replace("$num_books$", str(book_count)).replace("$num_hats$", str(hat_count)).replace("$num_balls$", str(ball_count))

        # add the values
        if "book_points" in prompt:

            if instance["agent"] == 0:
                book_value, hat_value, ball_value = instance["You_values"]["book"], instance["You_values"]["hat"], instance["You_values"]["ball"]
            else:
                book_value, hat_value, ball_value = instance["Them_values"]["book"], instance["Them_values"]["hat"], instance["Them_values"]["ball"]

            prompt = prompt.replace("$book_points$", str(book_value)).replace("$hat_points$", str(hat_value)).replace("$ball_points$", str(ball_value))

        return prompt

    def get_con_slot_prompt_dnd(self, instance, prev_instance, template):
        prompt = self.get_reg_slot_prompt_dnd(instance, template)

        if prev_instance:
            if prev_instance["agent"] == 0:
                prompt = prompt.replace("$previous_utterance$", "YOU: " + prev_instance["data"])
            elif prev_instance["agent"] == 1:
                prompt = prompt.replace("$previous_utterance$", "THEM: " + prev_instance["data"])
        else:
            prompt = prompt.replace("$previous_utterance$", "None")

        return prompt

    def get_reg_ul_prompt_cra(self, utterance, template):
        """Method to get a prompt given a template and utterance from the CRA
        dataset. This prompt does not provide previous utterances as context.

        Args:
            utterance: an utterance from a dialogue.
            template: the prompt template for a particular task.
        """
        prompt = template.replace("$utterance$", utterance)

        return prompt

    def get_con_ul_prompt_cra(self, utterance, context, template):
        """Method to get a prompt given a template and utterance from the CRA
        dataset. This prompt provides previous utterances as context, if available.

        Args:
            utterance: an utterance from a dialogue.
            context: previous utterances in the dialogue that facilitate understanding the current utterance.
            template: the prompt template for a particular task.
        """
        prompt = template.replace("$utterance$", utterance)

        if type(context) == str:
            # format this
            cxt = context.replace("\"", "").replace("\\n", " ").replace("\n", " ").split()
            cxt = [w for w in cxt if w != ""]
            print(cxt)
            cxt2 = []

            what = None
            for w in cxt:
                if w != "Alice:" and w != "Bob:":
                    cxt2.append(w)

                if w == "Alice:" or w == "Bob:":
                    if what == None or what != w:
                        cxt2.append(w)
                        what = w

            # only keep words for the last self.args.num_prior_utts utterances
            cxt2.reverse()
            cxt3 = []
            count = 0
            for w in cxt2:
                if w != "Alice:" and w != "Bob:":
                    cxt3.append(w)

                if w == "Alice:" or w == "Bob:":
                    cxt3.append(w)
                    count += 1

                if count == self.args.num_prior_utts:
                    break
            cxt3.reverse()
            cxt_str = " ".join(cxt3)

            prompt = prompt.replace("$previous_utterances$", cxt_str)
        else:
            prompt = prompt.replace("$previous_utterances$", "")

        # if context:
        #     prompt = prompt.replace("$previous_utterances$", context)
        # else:
        #     prompt = prompt.replace("$previous_utterances$", "None")

        # if "Alice: " in utterance:
        #     prompt = prompt.replace("$speaker$", "Alice")
        # elif "Bob: " in utterance:
        #     prompt = prompt.replace("$speaker$", "Bob")

        return prompt

    def make_con_ul_prompt_cra(self, instance, template):
        """Prepares to call the get_con_ul_prompt_cra method.

        Args:
            instance: a dict containing information about the speaker, context, and annotations for an utterance.
            template: the prompt template for a particular task.
        """

        if type(instance["context_str"]) != str:
                if instance["spkr"] == "A":
                    prompt = self.get_con_ul_prompt_cra("Alice: " + instance["utt"] + "\n", None, template)
                elif instance["spkr"] == "B":
                    prompt = self.get_con_ul_prompt_cra("Bob: " + instance["utt"] + "\n", None, template)
        else:
            if instance["spkr"] == "A":
                prompt = self.get_con_ul_prompt_cra("Alice: " + instance["utt"] + "\n", instance["context_str"], template)
            elif instance["spkr"] == "B":
                prompt = self.get_con_ul_prompt_cra("Bob: " + instance["utt"] + "\n", instance["context_str"], template)

        return prompt

    def get_prompt_ji(self, instance, template):
        """Method to get a prompt given a template and instance from the JobInterview
        dataset. This dialogue contains only the comments/messages between the agents.

        Args:
            instance: a Negotiation object that includes comments, proposed bids, and bids' acceptance/rejection.
            template: the prompt template for a particular task.
        """

        dialogue = ""
        for comment in instance.comments:
            dialogue += comment.user.context["role"] + ": " + comment.body + "\n"

        return template.replace("$dialogue$", dialogue)

    def get_da_dialogue_with_bids_ji(self, instance):
        """Method to get a dialogue from a JobInterview instance.

        Args:
            instance: a Negotiation object that includes comments, proposed bids, and bids' acceptance/rejection.
        """

        comments_dict = {}
        for comment in instance.comments:
            comment_str = comment.user.context["role"] + ": " + comment.body + "\n"
            comments_dict[comment.created_at] = comment_str

        bids_dict = {}
        for bid in instance.bids:
            bid_str = bid.user.context["role"] + ": < propose > " + str(bid.options) + "\n"

            if bid.accepted:
                if bid.user.context["role"] == "worker":
                    bid_response_str = "recruiter: < accept bid >" + "\n"
                elif bid.user.context["role"] == "recruiter":
                    bid_response_str = "worker: < accept bid >" + "\n"
            else:
                if bid.user.context["role"] == "worker":
                    bid_response_str = "recruiter: < reject bid >" + "\n"
                elif bid.user.context["role"] == "recruiter":
                    bid_response_str = "worker: < reject bid >" + "\n"

            bids_dict[bid.created_at] = [bid_str, bid_response_str]

        comments_dict.update(bids_dict)
        dialogue_dict = copy.deepcopy(comments_dict)
        list_of_tuples = sorted(dialogue_dict.items())

        dialogue = ""
        for timestamp, string_or_list in list_of_tuples:
            if type(string_or_list) == str:
                dialogue += string_or_list
            elif type(string_or_list) == list:
                dialogue += string_or_list[0] + string_or_list[1]

        return dialogue

    def get_dialogue_with_bids_ji(self, instance):
        """Returns a dialogue that is very similar to that output by get_da_dialogue_with_bids_ji, but does not include the conclusion of the dialogue (i.e. the acceptance/rejection of the final bid) if present.

        Args:
            instance: a Negotiation object that includes comments, proposed bids, and bids' acceptance/rejection.
        """

        full_dialogue = self.get_da_dialogue_with_bids_ji(instance)
        dialogue_lines = full_dialogue.split("\n")

        # remove empty lines
        dialogue_lines = [line.strip() for line in dialogue_lines if line.strip() != ""]

        # get rid of the conclusion so we can still ask the model whether the participants reached a deal
        if "< accept bid >" in dialogue_lines[-1] or "< reject bid >" in dialogue_lines[-1]:
            assert "< propose >" in dialogue_lines[-2]
            dialogue_lines_final = dialogue_lines[:-2]
        else:
            dialogue_lines_final = dialogue_lines[:]

        # check if we need to only use a partial dialogue
        if self.args.num_utts_partial_dial != -1:
            # use first k utterances only, from the start of the dialogue
            dialogue_lines_final = dialogue_lines_final[:self.args.num_utts_partial_dial]

        dialogue = ""
        for line in dialogue_lines_final:
            dialogue += line.strip() + "\n"

        return dialogue

    def get_prompt_with_bids_ji(self, instance, template):
        """Method to get a prompt given a template and instance from the JobInterview
        dataset.

        Args:
            instance: a Negotiation object that includes comments, proposed bids, and bids' acceptance/rejection.
            template: the prompt template for a particular task.
        """

        dialogue = self.get_dialogue_with_bids_ji(instance)

        temp_filled = template.replace("$dialogue$", dialogue)

        # fill in the values
        agent = "worker"
        if instance.users[0].context["role"] == agent:
            weights_list = {dict["name"]: dict["weight"] for dict in instance.users[0].context["utilities"]}
        elif instance.users[1].context["role"] == agent:
            weights_list = {dict["name"]: dict["weight"] for dict in instance.users[1].context["utilities"]}

        dict = copy.deepcopy(weights_list)

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

        temp_filled = temp_filled.replace("$pos_weight$", str(round(dict["position"], 3)))
        temp_filled = temp_filled.replace("$comp_weight$", str(round(dict["company"], 3)))
        temp_filled = temp_filled.replace("$salary_weight$", str(round(dict["salary"], 3)))
        temp_filled = temp_filled.replace("$workplace_weight$", str(round(dict["workplace"], 3)))
        temp_filled = temp_filled.replace("$days_off_weight$", str(round(dict["days_off"], 3)))

        return temp_filled

    def get_turns_from_dialogue_ji(self, instance):
        """Method to get the turns (a turn is a list of phrases) from a full dialogue in a JobInterview instance.

        Args:
            instance: a Negotiation object that includes comments, proposed bids, and bids' acceptance/rejection.
        """

        dialogue = self.get_da_dialogue_with_bids_ji(instance)
        dialogue_lines = dialogue.split("\n")
        turns_list = []
        if "recruiter:" in dialogue_lines[0]:
            speaker = "recruiter:"
        elif "worker:" in dialogue_lines[0]:
            speaker = "worker:"

        turn = []
        for index in range(len(dialogue_lines)):
            if "< propose >" in dialogue_lines[index] or "< accept bid >" in dialogue_lines[index] or "< reject bid >" in dialogue_lines[index]:    # any bid action is its own turn
                if turn:    # an ongoing turn is being recorded.
                    turns_list.append(turn)    # end the ongoing turn, put it in turns_list
                    turn = []    # start a new turn for the bid proposal
                    turn.append(dialogue_lines[index])    # put the bid proposal in the new turn
                    turns_list.append(turn)    # put the bid proposal turn in turns_list
                else:
                    turn.append(dialogue_lines[index])    # put the bid proposal in the empty, ongoing turn
                    turns_list.append(turn)    # put the bid proposal turn in turns_list

                turn = []    # start a new turn
                if index != len(dialogue_lines) - 1:
                    if "recruiter:" in dialogue_lines[index+1]:
                        speaker = "recruiter:"
                    elif "worker:" in dialogue_lines[index+1]:
                        speaker = "worker:"

            elif speaker in dialogue_lines[index]:
                turn.append(dialogue_lines[index])    # current speaker is saying something
            else:
                if turn:
                    turns_list.append(turn)    # opponent is saying something, so put the speaker's turn in
                    turn = []    # start a new turn for new speaker

                turn.append(dialogue_lines[index])    # put the new speaker's line in the new turn
                if speaker == "worker:":    # switch the speaker
                    speaker = "recruiter:"
                elif speaker == "recruiter:":
                    speaker = "worker:"

        return turns_list

    def get_all_turns(self, dataset_handler):
        """Get all turns present in the dialogues in the dataset.

        Args:
            dataset_handler: the dataset handler.
        """

        instances = dataset_handler.get_instances()
        num_dialogue = 0

        organized_turns = []
        for instance in instances:
            turns = self.get_turns_from_dialogue_ji(instance)

            for turn in turns:
                organized_turns.append({"num_dialogue": num_dialogue, "turn_comments": turn, "orig_instance": copy.deepcopy(instance)})

            num_dialogue += 1

        return organized_turns

    def get_turns(self, dataset_handler):
        """Get the turns from dialogues in the dataset.

        Args:
            dataset_handler: the dataset handler.
        """

        organized_turns = self.get_all_turns(dataset_handler)

        return organized_turns[:self.args.num_instances]

    def get_reg_da_prompt_ji(self, turn, template):
        turn_string = ""
        list_of_phrases = turn["turn_comments"]

        if "worker: " in list_of_phrases[0]:
            speaker = "worker: "
        elif "recruiter: " in turn["turn_comments"][0]:
            speaker = "recruiter: "

        for phrase in list_of_phrases:
            turn_string += phrase.replace("worker: ", "").replace("recruiter: ", "") + "\n"

        turn_string = speaker + turn_string
        temp_filled = template.replace("$utterance$", turn_string)

        # fill in the values
        agent = "worker"
        instance = turn["orig_instance"]
        if instance.users[0].context["role"] == agent:
            weights_list = {dict["name"]: dict["weight"] for dict in instance.users[0].context["utilities"]}
        elif instance.users[1].context["role"] == agent:
            weights_list = {dict["name"]: dict["weight"] for dict in instance.users[1].context["utilities"]}

        dict = copy.deepcopy(weights_list)

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

        temp_filled = temp_filled.replace("$pos_weight$", str(round(dict["position"], 3)))
        temp_filled = temp_filled.replace("$comp_weight$", str(round(dict["company"], 3)))
        temp_filled = temp_filled.replace("$salary_weight$", str(round(dict["salary"], 3)))
        temp_filled = temp_filled.replace("$workplace_weight$", str(round(dict["workplace"], 3)))
        temp_filled = temp_filled.replace("$days_off_weight$", str(round(dict["days_off"], 3)))

        return temp_filled

    def get_con_da_prompt_ji(self, turn, prev_turn, template):
        base_prompt = self.get_reg_da_prompt_ji(turn, template)

        if prev_turn:
            prev_turn_string = ""
            prev_list_of_phrases = prev_turn["turn_comments"]

            if "worker: " in prev_list_of_phrases[0]:
                prev_speaker = "worker: "
            elif "recruiter: " in prev_list_of_phrases[0]:
                prev_speaker = "recruiter: "

            for phrase in prev_list_of_phrases:
                prev_turn_string += phrase.replace("worker: ", "").replace("recruiter: ", "") + "\n"

            prev_turn_string = prev_speaker + prev_turn_string
            prompt = base_prompt.replace("$previous_utterance$", prev_turn_string)
        else:
            prompt = base_prompt.replace("$previous_utterance$", "None")

        return prompt

    def get_turn_dialogues(self, dataset_handler):
        """Get dialogues from the dataset that are formatted in terms of turns, rather than comments."""

        organized_turns = self.get_all_turns(dataset_handler)

        dialogue_counter = 0

        dialogues = []
        dialogue = ""

        for turn in organized_turns:
            # turn is part of ongoing dialogue.
            if turn["num_dialogue"] == dialogue_counter:
                turn_comments = turn["turn_comments"]

                for index in range(len(turn_comments)):
                    if index == 0:
                        dialogue += turn_comments[index] + "  "
                    else:
                        dialogue += turn_comments[index].replace("worker: ", "").replace("recruiter: ", "") + "  "

                dialogue += "\n"
            # turn is part of new dialogue.
            else:
                dialogues.append(dialogue)
                dialogue_counter += 1

                dialogue = ""
                turn_comments = turn["turn_comments"]

                for index in range(len(turn_comments)):
                    if index == 0:
                        dialogue += turn_comments[index] + "  "
                    else:
                        dialogue += turn_comments[index].replace("worker: ", "").replace("recruiter: ", "") + "  "

                dialogue += "\n"

        dialogues.append(dialogue)

        return dialogues

    def get_da_dialogue_prompt_ji(self, dialogue, template):
        dialogue = dialogue.replace("'", "")
        prompt = template.replace("$dialogue$", dialogue)

        return prompt

    def get_predictions(self, outputs, possible_outputs, cot_bool):
        """Method to extract the predictions from the model's outputs.

        Args:
            outputs: a list of the model's outputs.
            possible_outputs: a list of possible outputs specific to the task.
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
        """
        raise NotImplementedError # see get_final_outputs below.
        # for when we use the api key
        pred = []
        for key in outputs:
            if cot_bool:
                start = outputs[key].find("<answer>") + len("<answer>")
                end = outputs[key].find("</answer>")
                answer = outputs[key][start:end]
            else:
                answer = outputs[key]

            # exact output from model is a possible output.
            if answer in possible_outputs:
                pred.append(answer)
            # a possible output is somewhere within the model's response.
            elif any([po in answer.replace("Agent 1", "a").replace("Agent 2", "a") for po in possible_outputs]):
                list_of_words = []
                for word in answer.replace("Agent 1", "a").replace("Agent 2", "a").split(" "):
                    flags_for_word = [po for po in possible_outputs if po in word]
                    if flags_for_word:
                        list_of_words.append(flags_for_word[-1])
                pred.append(list_of_words[-1])
            else:
                pred.append("Manual prediction extraction required.")
        # for when we don't use the api key
        # pred.append(random.choice(possible_outputs))
        return pred

    def get_final_outputs(self, outputs_dict, possible_outputs, prompts, ground_truth, cot_bool=False):
        """Method to extract the predictions from the model's outputs.

        Args:
            outputs_dict: a dict of the model's outputs
            possible_outputs: a list of possible outputs specific to the task
            prompts: deduplicated prompts sent to the final
            ground_truth: corresponding ground_truth for the prompts.
        """

        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):

            if prompt not in outputs_dict:
                # could not find the answer due to some reason
                continue

            final_prompts.append(prompt)
            final_ground_truth.append(gt)

            answer = outputs_dict[prompt]

            if cot_bool:
                if "<answer>" not in answer or "</answer>" not in answer:
                    final_predictions.append("Manual prediction extraction required.")
                    continue
                start = answer.find("<answer>") + len("<answer>")
                end = answer.find("</answer>")
                answer = answer[start:end]

            # exact output from model is a possible output.
            if answer in possible_outputs:
                final_predictions.append(answer)
            elif any([po in answer.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                # a possible output is somewhere within the model's response.
                list_of_words = []
                for word in answer.replace("YOU", "a").replace("THEM", "a").split(" "):
                    flags_for_word = [po for po in possible_outputs if po in word]
                    if flags_for_word:
                        list_of_words.append(flags_for_word[-1])

                final_predictions.append(list_of_words[-1])
            else:
                final_predictions.append("Manual prediction extraction required.")

        return final_prompts, final_predictions, final_ground_truth


    def get_final_outputs_ann(self, outputs_dict, labels, prompts, ground_truth):
        """Method to extract the predictions from the model's outputs.

        Args:
            outputs_dict: a dict of the model's outputs
            labels: a list of possible outputs specific to the task
            prompts: deduplicated prompts sent to the final
            ground_truth: corresponding ground_truth for the prompts.
        """

        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):

            if prompt not in outputs_dict:
                # could not find the answer due to some reason
                continue

            final_prompts.append(prompt)
            final_ground_truth.append(gt)

            answer = outputs_dict[prompt]

            this_utterance_pred = []
            for label in labels:
                if label in answer:
                    this_utterance_pred.append(label)

            if this_utterance_pred:
                final_predictions.append(this_utterance_pred)
            else:
                final_predictions.append('Manual prediction extraction required. If no labels are found, please replace this with the Python list ["none"]. Ensure the none is in double quotes.')

        return final_prompts, final_predictions, final_ground_truth


    def get_predictions_dict_kc(self, outputs, possible_keys, possible_outputs):
        """Method to extract the predictions from the model's outputs.

        Version created by KC to handle json outputs.
        """
        raise NotImplementedError # see get_final_outputs_dict below.
        preds = []

        for key in outputs:
            pred = {} # also a json dict

            start = outputs[key].find("<answer>") + len("<answer>")
            end = outputs[key].find("</answer>")
            answer = outputs[key][start:end]

            try:
                answer = answer.replace("\n", "").replace(" ", "")
                annotations = json.loads(answer)
                assert len(annotations) == len(possible_keys)
                for k, v in annotations.items():
                    assert k in possible_keys
                    v = str(v)

                    # extract output from the model output.
                    if v in possible_outputs:
                        pred[k] = v
                    elif any([po in v.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                        # a possible output is somewhere within the model's response.
                        list_of_words = []
                        for word in v.replace("YOU", "a").replace("THEM", "a").split(" "):
                            flags_for_word = [po for po in possible_outputs if po in word]
                            if flags_for_word:
                                list_of_words.append(flags_for_word[-1])
                        pred[k] = list_of_words[-1]
                    else:
                        raise ValueError

                preds.append(pred)
            except:
                preds.append("Manual prediction extraction required.")

        return preds

    def get_final_outputs_dict(self, outputs_dict, possible_keys, possible_outputs, prompts, ground_truth):
        """
        Version created by KC to handle json outputs.
        """

        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):

            if prompt not in outputs_dict:
                # could not find the answer due to some reason
                continue

            final_prompts.append(prompt)
            final_ground_truth.append(gt)

            pred = {} # also a json dict

            start = outputs_dict[prompt].find("<answer>") + len("<answer>")
            end = outputs_dict[prompt].find("</answer>")
            answer = outputs_dict[prompt][start:end]

            try:
                answer = answer.replace("\n", "").replace(" ", "")
                annotations = json.loads(answer)
                assert len(annotations) == len(possible_keys)
                for k, v in annotations.items():
                    assert k in possible_keys
                    v = str(v)

                    # extract output from the model output.
                    if v in possible_outputs:
                        pred[k] = v
                    elif any([po in v.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                        # a possible output is somewhere within the model's response.
                        list_of_words = []
                        for word in v.replace("YOU", "a").replace("THEM", "a").split(" "):
                            flags_for_word = [po for po in possible_outputs if po in word]
                            if flags_for_word:
                                list_of_words.append(flags_for_word[-1])
                        pred[k] = list_of_words[-1]
                    else:
                        raise ValueError

                final_predictions.append(pred)
            except:
                final_predictions.append("Manual prediction extraction required.")

        return final_prompts, final_predictions, final_ground_truth

    def get_full_ann_predictions(self, outputs, cot_bool):
        """Method to extract the predictions from the model's outputs for tasks involving full-dialogue annotation.

        Args:
            outputs: a list of the model's outputs.
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
        """

        pred = []
        for key in outputs:
            if cot_bool:
                start = outputs[key].find("<answer>") + len("<answer>")
                end = outputs[key].find("</answer>")
                answer = outputs[key][start:end]
            else:
                answer = outputs[key]

            output_in_code = ast.literal_eval(answer)
            if type(output_in_code[0]) == tuple:
                annotations = [tuple[1] for tuple in output_in_code]
            else:
                annotations = output_in_code
            pred.append(annotations)

        return pred

    def get_all_slots_predictions(self, outputs, cot_bool):
        """Method to extract the outputs from a model's output for all-slots tasks.

        Args:
            outputs: a list of the model's outputs.
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
        """

        pred = []
        for key in outputs:
            start = outputs[key].find("<answer>") + len("<answer>")
            end = outputs[key].find("</answer>")
            answer = outputs[key][start:end]

            if "{" in answer:
                pred.append(answer.replace("\n  ", "").replace("\n ", "").replace("\n", "").strip().replace('\"', "'").replace('"', "'").replace("'0'", "0").replace("'1'", "1").replace("'2'", "2").replace("'3'", "3").replace("'4'", "4").replace("'5'", "5").replace(",'", ", '").replace(": unk", ": 'unk'"))
            else:
                pred.append("Manual prediction extraction required.")

        return pred

    def get_ann_predictions(self, outputs, cot_bool, labels):
        """Method to extract the outputs as is from a model's output for single-utterance dialogue-act annotation tasks.

        Args:
            outputs: a list of the model's outputs.
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
            labels: the options for dialogue acts.
        """

        pred = []
        for key in outputs:
            if cot_bool:
                start = outputs[key].find("<answer>") + len("<answer>")
                end = outputs[key].find("</answer>")
                answer = outputs[key][start:end]

            else:
                answer = outputs[key]

            this_utterance_pred = []
            for label in labels:
                if label in answer:
                    this_utterance_pred.append(label)
                    # answer.replace(label, "")

            if this_utterance_pred:
                pred.append(this_utterance_pred)
            else:
                pred.append('Manual prediction extraction required. If no labels are found, please replace this with the Python list ["none"].  Ensure the none is in double quotes.')

        return pred

    def log_everything(self, stats, prompts, predictions, ground_truth, outputs_dict, dataset_handler, model_handler):
        """Store the results and outputs in a json file.
        """

        if type(ground_truth) == list and type(ground_truth[0]) == list and type(ground_truth[0][0]) == list:
            ground_truth = self.flatten(ground_truth)
            preds = self.flatten(preds)

        if model_handler.multishot:
            storage = {
                "ground truth": ground_truth[2:],
                "predictions": preds,
                "prompts": prompts,
                "outputs_dict": outputs_dict,
            }
        else:
            storage = {
                "stats": stats,
                "ground truth": ground_truth,
                "predictions": predictions,
                "prompts": prompts,
                "outputs_dict": outputs_dict,
            }

        mname = model_handler.name
        if model_handler.name == "hf_model":
            mname = model_handler.args.hf_model_str.replace("/", "_")
        elif model_handler.name == "open_ai":
            mname = model_handler.args.openai_model_str
        out_path = utils.get_output_path(self.args.storage_dir, dataset_handler.name, mname, self.name, self.args.num_instances, args=self.args)

        utils.write_json(storage, out_path)


    def remove_duplicates(self, prompts, ground_truth):
        """Remove duplicate prompts and only keep ground truth for the prompts that are kept.

        Args:
            prompts: the original prompts (i.e. prompts for all instances).
            ground_truth: the original ground truth (i.e. ground truth for all instances).
        """

        assert len(prompts) == len(ground_truth)
        list_of_tuples = zip(prompts, ground_truth)

        unique_tuples = []
        seen_first_elements = []

        for tuple in list_of_tuples:
            if tuple[0] not in seen_first_elements:
                unique_tuples.append(tuple)
                seen_first_elements.append(tuple[0])

        new_prompts = [tuple[0] for tuple in unique_tuples]
        new_ground_truth = [tuple[1] for tuple in unique_tuples]

        return new_prompts, new_ground_truth
