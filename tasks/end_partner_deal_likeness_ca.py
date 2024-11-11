"""
Task Question: How much does your partner like you?
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y
"""

from tasks.k_task import KBaseTaskHandler


class LikenessCAHandler(KBaseTaskHandler):
    """Handler for the CaSiNo Agreeability task of determining how much an agent liked their fellow negotiator."""

    possible_outputs = ["extremely_dislike", "slightly_dislike", "undecided", "slightly_like", "extremely_like"]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
            """

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, utterance_bool=False, dialogue_bool=False, cot_bool=model_handler.cot, full_dialogue_bool=True)
        prompt_template = base_template.replace("$question$", "How much do you think your partner likes you?").replace("$output_specification$", "Present your answer as one of the following multiple choice options. You must select an option.\nA: extremely_dislike\nB: slightly_dislike\nC: undecided\nD: slightly_like\nE: extremely_like")

        return prompt_template

    def base_ground_truth(self, instances, agent):
        """Determine how much the agent likes their opponent.

        Args:
            dataset_handler: the dataset handler.
            agent: the agent whose feelings we want to determine.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        # a list of strings such as "Slightly like" or "Extremely dislike" for each respective instance.
        feelings_strs = [i["participant_info"][agent]["outcomes"]["opponent_likeness"] for i in instances]

        def like_str2cat(str):
            """Convert a string such as "Slightly like" to a Likert scale-type number.

                """
            return str.lower().replace(" ", "_")
            # if str == "Extremely dislike":
            #     return "Dislikes"
            # elif str == "Slightly dislike":
            #     return "Dislikes"
            # elif str == "Undecided":
            #     return "Neutral"
            # elif str == "Slightly like":
            #     return "Likes"
            # else:
            #     return "Likes"

        # return base ground truth: a list of categories in str format (ranging from "Dislikes" to "Likes").
        return [like_str2cat(str) for str in feelings_strs]

    def a1_base_ground_truth(self, instances):
        """Determine how much Agent 1 liked Agent 2.
        """

        return self.base_ground_truth(instances, "mturk_agent_1")

    def a2_base_ground_truth(self, instances):
        """Determine how much Agent 2 liked Agent 1.
        """

        return self.base_ground_truth(instances, "mturk_agent_2")


class A1PartnerLikenessCAHandler(LikenessCAHandler):
    """Handler for the CaSiNo Agreeability task of determining how much Agent 1 likes Agent 2."""

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

        # get the instances and dialogues from the dataset
        instances = dataset_handler.get_instances()

        self.prompt_template = self.get_prompt_template(dataset_handler, model_handler)

        # self.prompt_template = prompt_template.replace("$agent$", "mturk_agent_1").replace("$partner$", "mturk_agent_2")

        prompts = []
        for instance in instances:
            prompt = self.get_prompt_ca(instance, self.prompt_template, "mturk_agent_1")
            # prompt = prompt.replace("mturk_agent_1", "Agent 1").replace("mturk_agent_2", "Agent 2")
            prompts.append(prompt)

        # get the ground truth for this task.
        ground_truth = self.a2_base_ground_truth(instances)

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
