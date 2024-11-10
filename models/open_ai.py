"""
OpenAI models - GPT and variants.
"""


import openai
import os
from tqdm import tqdm
from models.model import BaseModelHandler
import tiktoken
import numpy as np


openai_api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = openai_api_key

class OpenAIHandler(BaseModelHandler):
    """Handler for the OpenAI model."""

    multishot = False
    cot=False

    possible_models = ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]

    token_limit = 4096

    def setup_model(self):
        """Setup the model."""

        self.model = self.args.openai_model_str

    def check_prompt(self, prompt):
        """
        Checks whether the number of tokens in the prompt violates the token limit of a
        particular model
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

        list_of_tokens = encoding.encode(prompt)
        num_tokens = len(list_of_tokens)

        return num_tokens < self.token_limit, num_tokens

    def process_gt(self, ppt, gt):
        """Process the ground truth as appropriate."""
        if type(gt) == str:
            return gt

        if type(gt) == list:
            if "Present your answer as a comma-separated list of strategies" in ppt:
                # follow <answer>self-need, empathy</answer>
                return f"<answer>{', '.join(gt)}</answer>"
            elif "Present your answer as a Python list of the relevant options" in ppt:
                # follow "```python\ndialogue_acts = [\"acknowledge\"]\n```",
                return f"```python\ndialogue_acts = {gt}\n```"
            else:
                raise ValueError
        raise ValueError

    def get_multishot_exs(self, inputs, ground_truth, index):
        """Get two multishot examples. process the ground_truth into a string appropriately."""
        assert len(inputs) == len(ground_truth)

        # choose two random indices apart from index
        all_ixs = list(range(len(inputs)))
        all_ixs.remove(index)
        ix1, ix2 = np.random.choice(all_ixs, 2, replace=False)

        ex_prompt, ex_ans, sec_ex_prompt, sec_ex_ans = inputs[ix1], ground_truth[ix1], inputs[ix2], ground_truth[ix2]

        # process the ground truth into a string.
        ex_ans_proc = self.process_gt(ex_prompt, ex_ans)
        sec_ex_ans_proc = self.process_gt(sec_ex_prompt, sec_ex_ans)

        return ex_prompt, ex_ans_proc, sec_ex_prompt, sec_ex_ans_proc

    def get_model_outputs(self, inputs, ground_truth):
        """Get the model outputs.

        Args:
            inputs: list of prompts to be passed to the model.
            ground_truth: the ground truth for the task.
        """

        outputs = {}

        if self.model in self.possible_models:
            if self.multishot:
                ex_prompt = inputs[0]
                sec_ex_prompt = inputs[1]

                if type(ground_truth[0]) == str:
                    ex_ans = ground_truth[0]
                else:
                    ex_ans = str(ground_truth[0])

                if type(ground_truth[1]) == str:
                    sec_ex_ans = ground_truth[1]
                else:
                    sec_ex_ans = str(ground_truth[1])

                for index in range(2, len(inputs)):
                    prompt = "User:\n" + ex_prompt + "\n\nAssistant: " + ex_ans + "\n\nUser:\n" + sec_ex_prompt + "\n\nAssistant: " + sec_ex_ans + "\n\nUser:\n" + inputs[index] + "\n\nAssistant:"

                    assert self.check_prompt(prompt)

                    gen_output = openai.ChatCompletion.create(
                        model = self.model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature = 0
                    )

                    output_text = gen_output["choices"][0]["message"]["content"]
                    outputs[prompt] = output_text
            else:
                for index in tqdm(range(len(inputs))):

                    if self.args.num_multishot == 0:
                        prompt = "User:\n" + inputs[index] + "\n\nAssistant:"
                    else:
                        assert self.args.num_multishot == 2
                        ex_prompt, ex_ans, sec_ex_prompt, sec_ex_ans = self.get_multishot_exs(inputs, ground_truth, index)
                        prompt = "User:\n" + ex_prompt + "\n\nAssistant: " + ex_ans + "\n\nUser:\n" + sec_ex_prompt + "\n\nAssistant: " + sec_ex_ans + "\n\nUser:\n" + inputs[index] + "\n\nAssistant:"

                        if index <= 2:
                            try:
                                print(f"Prompt at {index}: {prompt}")
                            except:
                                # print error
                                print(f"Error printing prompt at {index}.")
                                continue

                    outputs[inputs[index]] = "looks good."

                    # try:
                    #     a, b = self.check_prompt(prompt)
                    #     if not a:
                    #         print(f"Length issue at {index}/{len(inputs)}: {b} > {self.token_limit}")
                    #         continue

                    #     # outputs[inputs[index]] = "looks good"

                    #     gen_output = openai.ChatCompletion.create(
                    #         model = self.model,
                    #         messages = [{"role": "user", "content": prompt}],
                    #         temperature = 0
                    #     )

                    #     output_text = gen_output["choices"][0]["message"]["content"]
                    #     outputs[inputs[index]] = output_text
                    # except:
                    #     print(f"Some error here.")
                    #     continue

                    if (index + 1) >= self.args.max_num_instances:
                        break

        return outputs
