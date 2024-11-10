"""
CaSiNo dataset: https://huggingface.co/datasets/casino
"""


import ast
import copy
import pandas as pd
from nego_datasets.dataset import BaseDatasetHandler


class CasinoHandler(BaseDatasetHandler):
    """Handler for the CampSiteNegotiations (CaSiNo) dataset."""

    def setup_dataset(self):
        """Setup the dataset.

        Load the data from Huggingface. Do not use any randomization like shuffling here to ensure that the same instances are used for all evaluations.
        """

        df = pd.read_csv("./storage/utilities/ca.test.csv")
        ca_sample = df.to_dict(orient="records")

        self.dataset = []
        for inst in ca_sample:

            if "Walk-Away" not in inst["chat_logs"]:

                inst2 = copy.deepcopy(inst)

                inst2['annotations'] = ast.literal_eval(inst['annotations'])
                inst2['chat_logs'] = ast.literal_eval(inst['chat_logs'])
                inst2['participant_info'] = ast.literal_eval(inst['participant_info'])

                self.dataset.append(inst2)

    def get_instances(self):
        """Get the instances from the dataset."""

        # first k instances from the dataset
        instances = self.dataset[:self.args.num_instances]

        return instances

    def get_dial_template(self, counts_bool, values_bool, utterance_bool, dialogue_bool, cot_bool, full_dialogue_bool=False):
        """Get the prompt templates for CaSiNo dialogue comprehension tasks.

        Args:
            counts_bool: a boolean for whether the prompt provides item counts.
            values_bool: a boolean for whether the prompt provides item points.
            utterance_bool: a boolean for whether the prompt includes negotiation strategies.
            dialogue_bool: a boolean for whether the prompt provides dialogue.
            cot_bool: a boolean for whether chain of thought prompting is in use.
        """

        template = """Task Description: You are negotiating with your campsite neighbor over extra supply of food, water, and firewood for your camping trip. Different types of packages are worth different amount of points to each one of you. You'll be provided with information about the negotiation. Then, you'll answer a question."""

        if counts_bool:
            template += """\n\nHere are the number of food, water, and firewood packages available in the negotiation, contained in <count> tags.\n<count>\nFood Packages: 3\nWater Packages: 3\nFirewood Packages: 3\n</count>"""

        if values_bool:
            template += """\n\nHere are the number of points you get for each type of package, contained in <value> tags.\n<value>\nEach Food Package: $food_points$ points\nEach Water Package: $water_points$ points\nEach Firewood Package: $fire_points$ points\n</value>"""

        if utterance_bool:
            template += """Here are negotiation strategies and their definitions, contained in <strategy> tags: \n\n<strategy>\nsmall-talk: agent discusses topics apart from negotiation to build rapport\nempathy: agent demonstrates empathetic behavior toward the other agent\npromote-coordination: agent makes concession or expresses desire to find a deal\nno-need: agent states that they do not need an item based on personal context\nelicit-pref: agent attempts to gather the opponent's preferences\nundervalue-partner: agent undermines the needs of the opponent\nvouch-fairness: agent encourages fairness to benefit self\nself-need: agent creates arguments expressing why they need an item\nother-need: agent explains why someone in their party needs an item\n</strategy>"""

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

    def get_utt_template(self, counts_bool, values_bool, utterance_bool, cot_bool):
        """Get the prompt templates for CaSiNo dialogue comprehension tasks.

        Args:
            counts_bool: a boolean for whether the prompt provides item counts.
            values_bool: a boolean for whether the prompt provides item points.
            utterance_bool: a boolean for whether the prompt includes negotiation strategies.
            dialogue_bool: a boolean for whether the prompt provides dialogue.
            cot_bool: a boolean for whether chain of thought prompting is in use.
        """

        template = """Task Description: Task Description: You are negotiating with your campsite neighbor over extra supply of food, water, and firewood for your camping trip. Different types of packages are worth different amount of points to each one of you. You'll be provided with information about the negotiation. Then, you'll answer a question."""

        if counts_bool:
            template += """\n\nHere are the number of food, water, and firewood packages available in the negotiation, contained in <count> tags.\n<count>\nFood Packages: 3\nWater Packages: 3\nFirewood Packages: 3\n</count>"""

        if values_bool:
            template += """\n\nHere are the number of points you get for each type of package, contained in <value> tags.\n<value>\nEach Food Package: $food_points$ points\nEach Water Package: $water_points$ points\nEach Firewood Package: $fire_points$ points\n</value>"""

        if utterance_bool:
            template += """Here are the negotiation strategies and their definitions, contained in <strategy> tags: \n\n<strategy>\nsmall-talk: engaging in small-talk\nempathy: showing empathetic behavior towards the partner\ncoordination: making concessions or expressing desire to coordinate\nno-need: implying that a particular item is not required\nelicit-pref: inquiring about the partner's preferences\nundervalue-partner: undermining the needs of the partner\nvouch-fairness: encouraging fairness in the deal\nself-need: implying that a particular item is needed for personal use\nother-need: implying that a particular item is needed for someone else\n</strategy>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterance$\n</context>"""

        template += """\n\nHere is an utterance from the negotiation, contained in <utterance> tags:\n\n<utterance>\n$utterance$\n</utterance>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            # for simple chain-of-thought prompting.
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template
