"""
CRA dataset is created as part of the work found here: https://www.researchgate.net/publication/295854474_Toward_Natural_Turn-Taking_in_a_Virtual_Human_Negotiation_Agent

The code used to parse the raw data was written by Eleanor Lin and Kushal Chawla. The code used to create the .csv file was assisted by ChatGPT.
"""


from nego_datasets.dataset import BaseDatasetHandler
import os
import pandas as pd
import copy


class CRAHandler(BaseDatasetHandler):
    """Handler for the CRA dataset."""

    def setup_dataset(self):
        """Setup the dataset.

        Load the data from the corresponding .csv file. Do not use any randomization like shuffling here to ensure that the same instances are used for all evaluations.
        """
        # The csv was created with the help of Eleanor Lin and ChatGPT.

        # get cra sample from csv
        df = pd.read_csv(os.path.join("storage/", "utilities", "cra.test.csv"))

        # Convert the DataFrame to a list of dictionaries
        list_of_dicts = df.to_dict(orient='records')
        self.dataset = []

        for item in list_of_dicts:
            item2 = copy.deepcopy(item)
            item2["DUD"] = item2["DUD"].replace("\"", '')
            item2["DIV"] = item2["DIV"].replace("\"", '')
            item2["spkr"] = item2["spkr"].replace("\"", '')
            self.dataset.append(item2)

    def get_instances(self):
        """Get the instances from the dataset."""

        # first 10 instances from the dataset

        return self.dataset

    # this ground truth function is kept here because it is also used to get instances.
    def get_ground_truth(self):
        """Get the ground truth for this task."""

        instances = self.get_instances()

        ground_truth = []
        for instance in instances:
            instance_da = []

            if instance["make_offer"] == 1:
                instance_da.append("make offer")
            if instance["ask_offer"] == 1:
                instance_da.append("ask offer")
            if instance["accept"] == 1:
                instance_da.append("accept")
            if instance["reject"] == 1:
                instance_da.append("reject")
            if instance["ask_preference"] == 1:
                instance_da.append("ask preference")
            if instance["share_preference"] == 1:
                instance_da.append("share preference")
            if instance["make_offer"] != 1 and instance["ask_offer"] != 1 and instance["accept"] != 1 and instance["reject"] != 1 and instance["ask_preference"] != 1 and instance["share_preference"] != 1:
                instance_da.append("none")

            ground_truth.append(instance_da)

        # for instance in instances:
        #     if "Conventional-opening" in instance["GDA"]:
        #         ground_truth.append("greet")
        #     elif "Wh-Question" in instance["GDA"] or "Yes-no-question" in instance["GDA"] or instance["ask_offer"] == 1 or instance["ask_preference"] == 1 or "Open-question" in instance["GDA"]:
        #         ground_truth.append("inquire")
        #     elif "Action-directive" in instance["GDA"] or instance["make_offer"] == 1:
        #         ground_truth.append("propose")
        #     elif "Acknowledge-Agree-Accept-Yes-answers" in instance["GDA"] or "accept-deal" in instance["NDA"] or "accept-partial-deal" in instance["NDA"] or instance["accept"] == 1:
        #         ground_truth.append("agree")
        #     elif "No-answers" in instance["GDA"] or "reject-offer" in instance["NDA"] or "reject-negotiation" in instance["NDA"] or instance["reject"] == 1:
        #         ground_truth.append("disagree")
        #     elif "Collaborative-completion" in instance["GDA"] or "summarize-scenario" in instance["NDA"] or instance["share_preference"] == 1:
        #         ground_truth.append("inform")
        #     else:
        #         ground_truth.append("none")

        return ground_truth

    def get_da_instances(self):
        """Get instances with utterances whose dialogue acts are one of [greet, inquire, propose, agree, disagree, inform]."""

        instances = self.get_instances()
        ground_truth = self.get_ground_truth()
        assert len(instances) == len(ground_truth)

        da_instances = []
        for index in range(len(instances)):
            if ground_truth[index] != ["none"]:
                da_instances.append(instances[index])

        return da_instances[:self.args.num_instances]

    # this ground truth function is kept here because it is also used to get instances.
    def get_da_ground_truth(self):
        """Get the ground truth for instances with relevant dialogue acts (i.e. output by self.get_da_instances())."""

        ground_truth = self.get_ground_truth()

        da_ground_truth = [g for g in ground_truth if g != ["none"]]

        return da_ground_truth[:self.args.num_instances]

    def get_slot_instances(self):
        """Get the instances ground truth for instances that involve annotated proposals."""

        instances = self.get_instances()
        # ground_truth = self.get_ground_truth()
        # assert len(instances) == len(ground_truth)
        slot_instances = []
        for instance in instances:
            if instance["DIV"] != "[]" and ":" not in instance["DIV"] and instance["DUD"] != "[]":
                slot_instances.append(instance)
        return slot_instances[:self.args.num_instances]

    def get_utt_template(self, context_bool, da_bool, cot_bool):
        """Get the prompt templates for CRA utterance annotation tasks.

        Args:
            context_bool: a boolean for whether the prompt provides context for the utterance.
            da_bool: a boolean for whether the prompt asks for dialogue-act annotation.
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
        """

        template = "Task Description: You are negotiating with a partner over 1 painting, 2 lamps, and 3 records to determine who gets which items. Different types of items are worth different amount of points to each one of you. You'll be provided with an utterance from the conversation. Then, you'll answer a question."

        if da_bool:
            template += """\n\nHere is a list of dialogue acts, contained in <da> tags:\n\n<da>\nmake offer: proposing a full or a partial offer\nask offer: asking the partner to make a full or partial offer\naccept: agreeing to a previous offer\nreject: declining a previous offer\nask preference: asking the partner about which items they prefer\nshare preference: sharing which items you prefer\n</da>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterances$\n</context>"""

        template += """\n\nHere is an utterance, contained in <utterance> tags.\n<utterance>\n$utterance$\n</utterance>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            # for simple chain-of-thought prompting.
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template
