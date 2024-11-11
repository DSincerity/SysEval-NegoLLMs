"""
Base class for all tasks or tests.
"""


import copy
import utils
import json
import os

class WBaseTaskHandler:
    """Base handler for every task."""

    def __init__(self, name, args):
        """Initialize the task handler."""
        self.name = name
        self.args = args

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        """Primary method to evaluate a task on a given model and dataset.
        Stores instances, prompts, outputs, and ground truth.

        Args:
            dataset_handler: The dataset handler.
            model_handler: The model handler.
        """
        raise NotImplementedError

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

        # convert dict to a string, and concatenate with existing dialogue
        for log in logs:
            if log['text'] in ["Submit-Deal", "Accept-Deal", "Walk-Away", "Reject-Deal"]:
                # remove logistic utterances.
                continue
            round = log['id'] + ": " + log['text'] + "\n"
            dialogue += round

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

    def get_prompt_ns_ca(self, instance, template):
        """Method to get a prompt given a single instance from the casino
        dataset.
        Args:
            instance: a list of dicts containing chat_logs.
            template: the prompt template for a particular task

        Used in:
        - nego_strat_full_dial_single_ca.py
        - nego_strat_full_dial_ca.py
        """
        dialogue = ""
        logs = instance['chat_logs']
        participant_info = instance['participant_info']

        # convert dict to a string, and concatenate with existing dialogue
        for log in logs:
            if log['text'] != "Submit-Deal" and log['text'] != "Accept-Deal":
                if log['id'] == 'mturk_agent_1':
                    r = "Agent 1 : " + log['text'] + " \n"
                else:
                    r = "Agent 2 : " + log['text'] + " \n"
                dialogue += r

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

        agent1_value = [agent1_points[key] for key in ["Food", "Water", "Firewood"]]
        agent2_value = [agent2_points[key] for key in ["Food", "Water", "Firewood"]]

        prompt = template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$agent1_value$", str(agent1_value))
        prompt = prompt.replace("$agent2_value$", str(agent2_value))

        return prompt

    def get_prompt_dnd(self, instance, template, agent):
        """Method to get a prompt given a template and instance from the dnd
        dataset.

        Args:
            instance: a dict containing a row from the dataset
            template: the prompt template for a particular task
        """
        dialogue = ""
        dialogue_list = str(instance['dialogue']).split(" <eos> ")

        # skip the last selection utterance.
        for turn in dialogue_list[:-1]:
            dialogue += turn + "\n"

        # Update - let's just go ahead with YOU and THEM for now.
        # dialogue = dialogue.replace("YOU:", "Agent 1: ")
        # dialogue = dialogue.replace("THEM:", "Agent 2: ")

        you_value = instance['input']['value']
        them_value = instance['partner_input']['value']
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

    def get_prompt_da_dnd(self, instance, template):
        """
        Method to get a prompt given a template and instance from the dnd
        dataset for dialogue act tasks.

        Args:
            instance: a dict containing a row from the dataset
            template: the prompt template for a particular task

        Used in:
        - dial_act_full_dial_dnd.py
        """
        dialogue =""
        for i in range(len(instance['events'])):
            # pprint.pprint(instance['events'][i])
            if type(instance['events'][i]['data']) != dict:
                if instance['events'][i]['agent'] == 0:
                    dialogue += "Agent 1: " + instance['events'][i]['data'] + "\n"
                else:
                    dialogue += "Agent 2: " + instance['events'][i]['data'] + "\n"

        prompt = template.replace("$dialogue$", dialogue)
        return prompt

    def get_prompt_with_bids_ji(self, instance, template):
        """Method to get a prompt given a template and instance from the JI
        dataset.
        Args:
            instance: a dict containing a row from the dataset
            template: the prompt template for a particular task
        """
        comments_dict = {}
        for comment in instance.comments:
            comment_str = comment.user.context["role"] + ": " + comment.body + "\n"
            comments_dict[comment.created_at] = comment_str
        bids_dict = {}
        for bid in instance.bids:
            bid_str = bid.user.context["role"] + ": < propose > " + str(bid.options) + "\n"

            # from tara's code
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

        full_dialogue = ""
        for timestamp, string_or_list in list_of_tuples:
            if type(string_or_list) == str:
                full_dialogue += string_or_list
            elif type(string_or_list) == list:
                full_dialogue += string_or_list[0] + string_or_list[1]

        # from tara's code - remove final proposal and response
        dialogue_lines = full_dialogue.split("\n")

        # remove empty lines
        dialogue_lines = [line.strip() for line in dialogue_lines if line.strip() != ""]

        # get rid of the conclusion so we can still ask the model whether the participants reached a deal
        if "< accept bid >" in dialogue_lines[-1] or "< reject bid >" in dialogue_lines[-1]:
            assert "< propose >" in dialogue_lines[-2]
            dialogue_lines_final = dialogue_lines[:-2]
        else:
            dialogue_lines_final = dialogue_lines[:]

        dialogue = ""
        for line in dialogue_lines_final:
            dialogue += line + "\n"

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

    def log_everything(self, stats, prompts, predictions, ground_truth, outputs_dict, dataset_handler, model_handler):
        """Store the results and outputs in a json file."""

        if model_handler.multishot:
            storage = {
                "ground truth": ground_truth[2:],
                "predictions": predictions,
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

        #TEMP
        log_dir= "./"
        #out_path = utils.get_output_path(os.path.join(self.args.storage_dir, dataset_handler.name, mname,
        #                                 self.name, self.args.num_instances, args=self.args)
        out_path = utils.get_output_path(os.path.join(self.args.storage_dir,log_dir), dataset_handler.name, mname,
                                         self.name, self.args.num_instances, args=self.args)

        utils.write_json(storage, out_path)

    def log_everything_ut(self, instances, prompts, outputs, ground_truth, dataset_handler, model_handler):
        """Store the results and outputs in a json file."""

        if model_handler.multishot:
            storage = {
                "ground truth": ground_truth[2:],
                "instances": instances,
                "prompts": prompts,
                "outputs": outputs,
            }
        else:
            storage = {
                "ground truth": ground_truth,
                "instances": instances,
                "prompts": prompts,
                "outputs": outputs,
            }

        out_path = utils.get_output_path(self.args.storage_dir, dataset_handler.name, model_handler.name, self.name, self.args.num_instances, self.args.num_utts_partial_dial)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        utils.write_json(storage, out_path)


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
                # could not find the answer due to some reason such as token length issue.
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



    def get_partial_dial_ca(self, num_utt, instance, prompt_template):
        """
        Method to get a partial dialogue given a single instance from the casino, the
        number of utterances to return, and a prompt template.

        Args:
            num_utt: the number of utterances for the partial dialogue to contain
            instance: a dict containing a row from the dataset
            prompt_template: the prompt template for a particular task

        Used in:
        - gen_single_ca.py
        """
        dialogue = ""
        logs = instance['chat_logs']
        participant_info = instance['participant_info']

        # max past 5 utterances - recent history before the num_utt utterance.
        history = logs[:num_utt]

        # remove special utterances
        history2 = []
        for utt in history:
            if utt['text'] not in ["Submit-Deal", "Accept-Deal", "Walk-Away", "Reject-Deal"]:
                history2.append(utt)

        # use only 5 most recent
        if len(history2) > 5:
            history2 = history2[-5:]

        for utt in history2:
            dialogue += utt['id'] + ": " + utt['text'] + "\n"

        # use you and them - same as DND. tread mturk_agent_1 as YOU
        dialogue = dialogue.replace("mturk_agent_1:", "YOU:")
        dialogue = dialogue.replace("mturk_agent_2:", "THEM:")

        # a priority dict in the form {'Low': 'Water', 'Medium': 'Food', 'High': 'Firewood'}.
        agent1_dict = participant_info["mturk_agent_1"]["value2issue"]

        # get dicts in the form {'Water': 'Low', 'Food': 'Medium', 'Firewood': 'High'}.
        agent1_switched = {item: level for level, item in agent1_dict.items()}

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

        prompt = prompt_template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$food_points$", str(agent1_points['Food']))
        prompt = prompt.replace("$water_points$", str(agent1_points['Water']))
        prompt = prompt.replace("$fire_points$", str(agent1_points['Firewood']))

        return prompt

    def get_partial_dial_dnd(self, num_utt, instance, dialogue_list, prompt_template):
        """
        Method to get a partial dialogue given a single instance from the casino, the
        number of utterances to return, and a prompt template.

        Args:
            num_utt: the number of utterances for the partial dialogue to contain
            instance: a dict containing a row from the dataset
            prompt_template: the prompt template for a particular task
        """
        dialogue = ""
        #use only past 5 utterances in the history before the current utterance given by num_utt

        history = dialogue_list[:num_utt] #recent history before the num_utt utterance.
        # use only 5 most recent
        if len(history) > 5:
            history = history[-5:]
        for utt in history:
            dialogue += utt.strip() + "\n"

        # always use your own values.
        value = instance['input']['value']

        counts = instance['input']['count']

        books = str(counts[0])
        hats = str(counts[1])
        balls = str(counts[2])

        book_pts = str(value[0])
        hat_pts = str(value[1])
        ball_pts = str(value[2])

        prompt = prompt_template.replace("$dialogue$", dialogue)
        # prompt = prompt.replace("$agent_name$", resp)
        prompt = prompt.replace("$num_books$", books)
        prompt = prompt.replace("$num_hats$", hats)
        prompt = prompt.replace("$num_balls$", balls)
        prompt = prompt.replace("$book_points$", book_pts)
        prompt = prompt.replace("$hat_points$", hat_pts)
        prompt = prompt.replace("$ball_points$", ball_pts)

        return prompt

    def remove_duplicates(self, prompts, ground_truth):
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
