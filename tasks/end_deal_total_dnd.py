"""
Task Question: How many total points did you get at the end of the negotiation?
Info:
    - Item Counts: Y
    - Self Points: Y
    - Dialogue: Y
"""


from tasks.k_task import KBaseTaskHandler


class DNDDealPointsHandler(KBaseTaskHandler):
    """Handler for the DealOrNoDeal Deal Points task of determining how many points each agent achieved once the agents reached a deal (diagreement/agreement to walk away is classified as a deal to achieve 0 points each)."""

    possible_outputs = [str(num) for num in range(-40, 40)]

    def get_prompt_template(self, dataset_handler, model_handler):
        """Get the basic prompt template for the task, using functions from the dataset handler.

        Args:
            dataset_handler: the dataset handler.
            model_handler: the model handler.
        """

        base_template = dataset_handler.get_dial_template(counts_bool=True, values_bool=True, dialogue_bool=False, da_bool=False, cot_bool=model_handler.args.use_cot, full_dialogue_bool=True)
        prompt_template = base_template.replace("$question$", "How many points did you get at the end of the negotiation?").replace("$output_specification$", "Present your answer as a single number with no additional text. If the agents did not reach a deal, answer 0.")

        return prompt_template

    def find_deal_counts(self, deal_lst):
        """Extract relevant information from the lists of strings describing deals in the dataset.

        Args:
            deal_lst: a list with strings describing the deal achieved by the agents.
        """

        if deal_lst[0] == "<disagree>" or deal_lst[0] == "<no_agreement>":
            new_lst = [0, 0, 0]
        else:
            new_lst = []
            for string in deal_lst:
                string_count = int(string[string.index("=") + 1:])
                new_lst.append(string_count)
        return new_lst

    def find_deal_points(self, count_lst, value_lst):
        """Use information about how many items an agent received to determine how many points they scored in the deal.

        Args:
            count_lst: a list of the counts of each item type an agent received.
            value_lst: a list describing how many points each item type is worth to the agent.
        """

        points_lst = []
        for index in range(len(count_lst)):
            points = count_lst[index] * value_lst[index]
            points_lst.append(points)
        return str(sum(points_lst))

    def you_base_ground_truth(self, instances):
        """Get the number of points Agent YOU achieved in the deal.

        Args:
            dataset_handler: The dataset handler.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        # a list of deal quantity lists in the form [you_deal_books, you_deal_hats, you_deal_balls, them_deal_books, them_deal_hats, them_deal_balls].
        deals = [i["output"].split(" ") for i in instances]

        # a list of Agent You's deal quantity lists in the form [you_deal_books, you_deal_hats, you_deal_balls].
        you_deals = [lst[:3] for lst in deals]

        # update you_deals to you_counts so that the quantity lists are simply lists of integers.
        you_counts = [self.find_deal_counts(lst) for lst in you_deals]

        # use Agent You's deal_counts from you_counts and value function to compute the points for an instance.
        # return the base ground truth: a list of ints representing the points scored by Agent You for each respective instance.
        return [self.find_deal_points(you_counts[index], instances[index]["input"]["value"]) for index in range(len(instances))]

    def them_base_ground_truth(self, instances):
        """Get the number of points Agent THEM achieved in the deal.

        Args:
            dataset_handler: The dataset handler.
        """

        # get the instances from the dataset.
        # instances = dataset_handler.get_instances()

        # a list of deal quantity lists in the form [you_deal_books, you_deal_hats, you_deal_balls, them_deal_books, them_deal_hats, them_deal_balls].
        deals = [i["output"].split(" ") for i in instances]

        # a list of Agent Them's deal quantity lists in the form [them_deal_books, them_deal_hats, them_deal_balls].
        them_deals = [lst[3:] for lst in deals]

        # update them_deals to them_counts so that the quantity lists are simply lists of integers.
        them_counts = [self.find_deal_counts(lst) for lst in them_deals]

        # use Agent Them's deal_counts from you_counts and value function to compute the points for an instance.
        # return the base ground truth: a list of ints representing the points scored by Agent Them for each respective instance.
        return [self.find_deal_points(them_counts[index], instances[index]["partner_input"]["value"]) for index in range(len(instances))]


class YDNDDealPointsHandler(DNDDealPointsHandler):
    """Handler for the DealOrNoDeal Deal Points task of determining how many points Agent YOU achieved in the deal."""

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

        # self.prompt_template = prompt_template.replace("$agent_name$", "YOU")

        prompts = []
        for instance in instances:
            prompt = self.get_prompt_dnd(instance, self.prompt_template, "YOU")
            # prompt = prompt.replace("YOU:", "Agent 1:").replace("THEM:", "Agent 2:").replace("Agent YOU", "Agent 1").replace("Agent THEM", "Agent 2")
            prompts.append(prompt)

        # get the ground truth for this task.
        ground_truth = self.you_base_ground_truth(instances)

        new_prompts, new_ground_truth = self.remove_duplicates(prompts, ground_truth)

        if return_prompt_gt:
            return new_prompts, new_ground_truth

        # get the model outputs - dict from prompt to the output. It's possible that some are missing so a dict is better than a list.
        outputs_dict = model_handler.get_model_outputs(new_prompts, new_ground_truth)

        #only for the ones that are unique and where valid predictions are available
        final_prompts, final_predictions, final_ground_truth = self.get_final_outputs(outputs_dict, self.possible_outputs, new_prompts, new_ground_truth, cot_bool=model_handler.args.use_cot)

        # log everything
        stats = {
            "total": len(prompts),
            "unique": len(new_prompts),
            "valid": len(final_prompts),
        }

        self.log_everything(stats, final_prompts, final_predictions, final_ground_truth, outputs_dict, dataset_handler, model_handler)

        return instances
