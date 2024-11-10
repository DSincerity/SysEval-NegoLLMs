import re


class PromptModifier(object):
    """
    Class for modifying prompt for different tasks

    """

    def __init__(self):
        pass

    def modify(self, prompt, task_nm):
        prompt = prompt.replace(
            "Task Description: ", self.map_prefix(task_nm)
        )  # Change prefix
        prompt = re.sub(
            "(Predict.+?: )(.*\s*)(Here)", "\\1\n\n\\3", prompt, flags=re.DOTALL
        )  # remove task descriptions
        prompt = re.sub("Here.*tags.", "", prompt)  # remove description of tags
        prompt = self.parts_of_delete_in_prompt(prompt, task_nm)
        prompt = re.sub(r"\n\n+\s*", "\n\n", prompt)  # newlines over 3 to 2
        return prompt

    def parts_of_delete_in_prompt(self, prompt, task_nm):
        if "end_deal_specifics" in task_nm:
            return self.value_part_delete(prompt)
        else:
            return prompt

    def value_part_delete(self, prompt):
        return re.sub(r"\<value\>.*?\<\/value\>", "", prompt, flags=re.DOTALL)

    def dialogue_part_delete(self, prompt):
        return re.sub(r"\<dialogue\>.*?\<\/dialogue\>", "", prompt, flags=re.DOTALL)

    def count_part_delete(self, prompt):
        return re.sub(r"\<count\>.*?\<\/count\>", "", prompt, flags=re.DOTALL)

    def map_prefix(self, task_nm):
        if "ask_high_priority" in task_nm and "partner" not in task_nm:
            prefix = "Predict high priority issue: "
        elif "ask_low_priority" in task_nm:
            prefix = "Predict low priority issue: "
        elif "partner_ask_high_priority" in task_nm:
            prefix = "Predict partner high priority issue: "
        elif "partner_ask_low_priority" in task_nm:
            prefix = "Predict partner low priority issue: "
        elif "likeness" in task_nm and "partner" not in task_nm:
            prefix = "Predict deal likeness: "
        elif "satisfaction" in task_nm and "partner" not in task_nm:
            prefix = "Predict deal satisfaction: "
        elif "likeness" in task_nm:
            prefix = "Predict partner deal likeness: "
        elif "satisfaction" in task_nm:
            prefix = "Predict partner deal satisfaction: "
        elif "dial_act" in task_nm:
            prefix = "Predict dialogue act: "
        elif "deal_achieved" in task_nm:
            prefix = "Predict whether deal was achieved: "
        elif "max_point" in task_nm:
            prefix = "Predict max point: "
        elif "full_proposal" in task_nm:
            prefix = "Predict full proposal: "
        elif "deal_total" in task_nm:
            prefix = "Predict deal total: "
        elif "deal_specific" in task_nm:
            prefix = "Predict deal specifics: "
        elif "strategy" in task_nm:
            prefix = "Predict strategy: "
        else:
            raise ValueError("task name is not correct")
        return prefix
