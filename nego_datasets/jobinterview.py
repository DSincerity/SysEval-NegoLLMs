"""
JobInterview dataset and managing code is part of gucci-j's GitHub repository: https://github.com/gucci-j/negotiation-breakdown-detection.
"""


import os
import pandas as pd
from nego_datasets.dataset import BaseDatasetHandler
from nego_datasets.negotiation_ji import read_ji_negotiations


class JIHandler(BaseDatasetHandler):
    """Handler for the JobInterview (JI) dataset."""

    def setup_dataset(self):
        """Setup the dataset.

        Load the data from a clone of gucci-j's Negotiation Breakdown Detection repository. Do not use any randomization like shuffling here to ensure that the same instances are used for all evaluations.
        """

        job_interview = read_ji_negotiations(os.path.join("storage/", "utilities", "ji.test.json"))

        self.dataset = job_interview

        # the rest of the code in this function was assisted by ChatGPT
        # read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join("storage/", "utilities", "ji_dacts.test.csv"))

        # convert the DataFrame to a list of dictionaries
        list_of_dicts = df.to_dict(orient='records')
        # list_of_dicts = [dict for dict in list_of_dicts if dict["flag"] == 0]
        self.da_list = list_of_dicts

        # FOR WHEN WE WANT TO ACTUALLY TEST ON THE SAMPLE SET.
        # self.dataset = [self.dataset[index] for index in ji_sample_indices]

        # self.da_list = [self.da_list[index] for index in ji_sample_indices]

    def get_instances(self):
        """Get the instances from the dataset."""

        # first k instances from the dataset
        instances = self.dataset[:self.args.num_instances]

        return instances

    def get_dial_template(self, counts_bool, cot_bool, values_bool=False, dialogue_bool=False, full_dialogue_bool=False):
        """Get the prompt templates for CaSiNo dialogue comprehension tasks.

        Args:
            counts_bool: a boolean for whether the prompt provides item counts.
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
            """

        template = "Task Description: You are a worker who is negotiating with a recruiter over the issues surrounding a job offer. There are 5 issues to discuss: position, company, salary, workplace, and weekly days off. You both value these issues differently. You'll be provided with information about the negotiation. Then, you'll answer a question."

        if counts_bool:
            template += " There are 4 options for position, 4 options for company, and 4 options for workplace. Salary ranges from $20 to $50, and the number of possible weekly days off ranges from 2 to 5."

        if values_bool:
            template += """\n\nHere are the weights that represent your preference towards each issue in <value> tags.\n<value>\nposition: $pos_weight$\ncompany: $comp_weight$\nsalary: $salary_weight$\nworkplace: $workplace_weight$\ndays_off: $days_off_weight$</value>"""

        if dialogue_bool:
            template += """\n\nHere is the recent dialogue history, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""

        if full_dialogue_bool:
            template += """\n\nHere is the complete dialogue, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            # for simple chain-of-thought prompting.
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template

    def get_utt_template(self, context_bool, full_dial_bool, cot_bool, counts_bool=False, values_bool=False, da_bool=False):
        """Get the prompt templates for CRA utterance annotation tasks.

        Args:
            context_bool: a boolean for whether the prompt provides context for the utterance.
            full_dial_bool: a boolean for whether the prompt presents the full dialogue for annotation (instead of just an utterance).
            cot_bool: a boolean for whether chain-of-thought prompting is in use.
        """

        template = "Task Description: You are a worker who is negotiating with a recruiter over the issues surrounding a job offer. There are 5 issues to discuss: position, company, salary, workplace, and weekly days off. You both value these issues differently. You'll be provided with information about the negotiation. Then, you'll answer a question."

        if counts_bool:
            template += " There are 4 options for position, 4 options for company, and 4 options for workplace. Salary ranges from $20 to $50, and the number of possible weekly days off ranges from 2 to 5."

        if values_bool:
            template += """\n\nHere are the weights that represent your preference towards each issue in <value> tags.\n<value>\nposition: $pos_weight$\ncompany: $comp_weight$\nsalary: $salary_weight$\nworkplace: $workplace_weight$\ndays_off: $days_off_weight$</value>"""

        if da_bool:
            template += """\n\nHere is a list of dialogue acts, contained in <da> tags:\n\n<da>\ngreet: greeting the partner.\ninquire: asking an open-ended question.\npropose: suggesting an offer or aspect of an offer.\nagree: agreeing to a previous offer.\ndisagree: declining a previous offer.\ninform: sharing useful information such as what they like\dislike the most.\nunknown: none of the dialogue acts above apply.\n</da>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterance$\n</context>"""

        if full_dial_bool:
            template += """\n\nHere is the agents' dialogue, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""
        else:
            template += """\n\nHere is an utterance, contained in <utterance> tags.\n<utterance>\n$utterance$\n</utterance>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            # for simple chain-of-thought prompting.
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template
