"""
DealorNoDeal dataset: https://huggingface.co/datasets/deal_or_no_dialog
"""


import json
import pandas as pd
from nego_datasets.dataset import BaseDatasetHandler


class DNDHandler(BaseDatasetHandler):
    """Handler for the DealOrNoDeal (DND) dataset."""

    def setup_dataset(self):
        """Setup the dataset.

        Load the data from Huggingface and from train-parsed.json. Do not use any randomization like shuffling here to ensure that the same instances are used for all evaluations.
        """

        df = pd.read_csv("storage/utilities/dnd.test.csv")
        dnd_sample = df.to_dict(orient="records")

        for inst in dnd_sample:
            inst['input'] = json.loads(inst['input'])
            inst['partner_input'] = json.loads(inst['partner_input'])

        self.dataset = dnd_sample

        df = pd.read_csv("storage/utilities/dnd_ann.test.csv")
        ann_sample = df.to_dict(orient="records")

        for inst in ann_sample:

            inst['agents'] = json.loads(inst['agents'])
            inst['agents_info'] = json.loads(inst['agents_info'])
            inst['events'] = json.loads(inst['events'])
            inst['outcome'] = json.loads(inst['outcome'])
            inst['scenario'] = json.loads(inst['scenario'])

        self.annotated_dataset = ann_sample

    def get_instances(self):
        """Get the instances from the dataset."""

        # first k instances from the dataset
        instances = self.dataset[:self.args.num_instances]

        return instances

    # Probably an error. Duplicate name.
    # def get_da_instances(self):
    #     """Get the instances from the annotated dataset."""

    #     # last 10 instances from the dataset
    #     instances = self.dataset[-self.args.num_instances:]

    #     return instances

    def get_da_instances(self):
        """Get the instances from the annotated dataset."""

        # first k instances from the dataset
        instances = self.annotated_dataset[:self.args.num_instances]

        return instances

    def get_annotated_utterances(self):
        """Get the utterances from the annotated dataset."""

        # utterances as is and "< sep >" between dialogues and no dialogue tags.
        organized_utterances = []
        for instance in self.annotated_dataset:
            for utterance_dict in instance["events"]:
                if utterance_dict["action"] == "message":
                    counts = {dict["Name"]: dict["Count"] for dict in instance["scenario"]["kbs"][0]}
                    you_values = {dict["Name"]: dict["Value"] for dict in instance["scenario"]["kbs"][0]}
                    them_values = {dict["Name"]: dict["Value"] for dict in instance["scenario"]["kbs"][1]}
                    utterance_dict["Counts"] = counts
                    utterance_dict["You_values"] = you_values
                    utterance_dict["Them_values"] = them_values
                    organized_utterances.append(utterance_dict)
            organized_utterances.append("< sep >")

        dialogue_num = 0

        # utterances with dialogue tags; "< sep >" is not not present in the list.
        with_dialogue_tags = []
        for utt in organized_utterances:
            if utt != "< sep >":
                utt["dialogue_num"] = dialogue_num
                with_dialogue_tags.append(utt)
            else:
                dialogue_num += 1

        # first k utterances from the dataset
        ann_utterances = with_dialogue_tags[:self.args.num_instances]

        return with_dialogue_tags, ann_utterances

    def get_propose_and_extra_utterances(self):
        """Get the utterances from the annotated dataset for which the dialogue act is "propose"."""

        with_dialogue_tags, _ = self.get_annotated_utterances()

        max_prop_utterances = 0
        for item in with_dialogue_tags:
            if item["metadata"]["intent"] == "propose":
                max_prop_utterances += 1

        # gets the correct number (self.args.num_instances) of propose utterances, but keeps the utterances in between these (for prompts that involve context).
        prop_utterances = []
        num_prop_utterances = 0
        index = 0

        while num_prop_utterances < min(max_prop_utterances, self.args.num_instances):
            utt = with_dialogue_tags[index]

            if utt["metadata"]["intent"] == "propose":
                num_prop_utterances += 1

            prop_utterances.append(utt)
            index += 1

        return prop_utterances

    def get_dial_template(self, counts_bool, values_bool, dialogue_bool, da_bool, cot_bool, full_dialogue_bool=False):
        """Get the prompt templates for CaSiNo dialogue comprehension tasks.

        Args:
            counts_bool: a boolean for whether the prompt provides item counts.
            values_bool: a boolean for whether the prompt provides item points.
            dialogue_bool: a boolean for whether the prompt provides dialogue.
            da_bool: a boolean for whether the prompt provides dialogue acts.
            cot_bool: a boolean for whether chain of thought prompting is in use.
        """

        template = """Task Description: You are negotiating with a partner over some quantity of books, hats, and balls to determine who gets which items. Different types of items are worth different amount of points to each one of you. You'll be provided with information about the negotiation. Then, you'll answer a question."""

        if counts_bool:
            template += """\n\nHere are the number of books, hats, and balls available in the negotiation, contained in <count> tags.\n<count>\nBooks: $num_books$\nHats: $num_hats$\nBalls: $num_balls$\n</count>"""

        if values_bool:
            template += """\n\nHere are the number of points you get for each type of item, contained in <value> tags.\n<value>\nEach Book: $book_points$ points\nEach Hat: $hat_points$ points\nEach Ball: $ball_points$ points\n</value>"""

        if da_bool:
            template += """\n\nHere are a list of dialogue acts, contained in <da> tags: \n\n<da>\ngreet\ninquire\npropose\nagree\ndisagree\ninsist\nunknown\n</da>"""

        if dialogue_bool:
            template += """\n\nHere is the recent dialogue history, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""

        if full_dialogue_bool:
            template += """\n\nHere is the complete dialogue, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            # for simple chain-of-thought prompting.
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer as a single number in <answer> </answer> tags."""
        else:
            template += " $output_specification$"

        return template

    def get_utt_template(self, counts_bool, values_bool, context_bool, da_bool, cot_bool):
        """Get the prompt templates for CRA utterance annotation tasks.

        Args:
            counts_bool: a boolean for whether the prompt provides counts.
            values_bool: a boolean for whether the prompt provides item points.
            context_bool: a boolean for whether the prompt provides context for the utterance.
            da_bool: a boolean for whether the prompt asks for dialogue-act annotation.
            cot_bool: a boolean for whether chain of thought prompting is in use.
        """

        template = """Task Description: You are negotiating with a partner over some quantity of books, hats, and balls to determine who gets which items. Different types of items are worth different amount of points to each one of you. You'll be provided with information about the negotiation. Then, you'll answer a question."""

        if counts_bool:
            template += """\n\nHere are the number of books, hats, and balls available in the negotiation, contained in <count> tags.\n<count>\nBooks: $num_books$\nHats: $num_hats$\nBalls: $num_balls$\n</count>"""

        if values_bool:
            template += """\n\nHere are the number of points you get for each type of item, contained in <value> tags.\n<value>\nEach Book: $book_points$ points\nEach Hat: $hat_points$ points\nEach Ball: $ball_points$ points\n</value>"""

        if da_bool:
            template += """\n\nHere are a list of dialogue acts, contained in <da> tags: \n\n<da>\ngreet\ninquire\npropose\nagree\ndisagree\ninsist\nunknown\n</da>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterance$\n</context>"""

        template += """\n\nHere is an utterance from the negotiation, contained in <utterance> tags.\n<utterance>\n$utterance$\n</utterance>"""

        template += "\n\nQuestion: $question$"

        if cot_bool:
            # for simple chain-of-thought prompting.
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer as a single number in <answer> </answer> tags."""
        else:
            template += " $output_specification$"

        return template
