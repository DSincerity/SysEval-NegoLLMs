"""
HF-based opensource models - Any model that is on the Huggingface hub - evaluated on USC CARC.
"""

import os
from models.model import BaseModelHandler
from tqdm import tqdm

class HFModelHandler(BaseModelHandler):
    """Handler for the OpenAI model."""

    multishot = False
    cot = False

    def setup_model(self):
        """Setup the model."""

        if "flan-t5" in self.args.hf_model_str:
            self.token_limit = 512
            self.max_new_tokens = 100
        elif "falcon" in self.args.hf_model_str:
            self.token_limit = 1500
            self.max_new_tokens = 2048
        elif "mistral" in self.args.hf_model_str:
            self.token_limit = 1500
            self.max_new_tokens = 200
        elif "vicuna" in self.args.hf_model_str or "Wizard" in self.args.hf_model_str:
            self.token_limit = 1500
            self.max_new_tokens = 200
        else:
            raise ValueError


        if "flan-t5" in self.args.hf_model_str:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.hf_model_str, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        elif "falcon" in self.args.hf_model_str:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_str, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        elif "mistral" in self.args.hf_model_str:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_str, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        elif "vicuna" in self.args.hf_model_str or "Wizard" in self.args.hf_model_str:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_str, device_map="auto", low_cpu_mem_usage=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        else:
            raise ValueError

    def check_prompt(self, prompt):
        """
        Checks whether the number of tokens in the prompt violates the token limit of a
        particular model.
        """

        if "mistral" in self.args.hf_model_str:
            inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids

        return inputs.shape[1] < self.token_limit, inputs.shape[1]

    def get_model_outputs(self, inputs, ground_truth):
        """Get the model outputs.

        Args:
            inputs: list of prompts to be passed to the model.
            ground_truth: the ground truth for the task.
        """

        outputs = {}

        if self.multishot or self.cot:
            raise NotImplementedError

        for index in tqdm(range(len(inputs))):

            # outputs[inputs[index]] = "looks good"

            # if len(outputs) >= self.args.max_num_instances:
            #     break

            if "flan-t5" in self.args.hf_model_str:
                prompt = inputs[index]
            elif "mistral" in self.args.hf_model_str:
                prompt = [
                    {"role": "user", "content": inputs[index]},

                ]
            else:
                prompt = "User: " + inputs[index] + " Assistant: "

                # for w in ["count", "value", "da", "utterance"]:
                #     prompt = prompt.replace(f", contained in <{w}> tags.\n<{w}>",":").replace(f"</{w}>","")

            a, b = self.check_prompt(prompt)
            if not a:
                print(f"Length issue at {index}/{len(inputs)}: {b} > {self.token_limit}")
                continue

            outputs[inputs[index]] = self.get_hf_model_output(prompt)

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs

    def get_hf_model_output(self, prompt):
        """Get the output from the HF model."""

        if "flan" in self.args.hf_model_str:
            # not supporting this anymore - has a small input length.
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        elif "falcon" in self.args.hf_model_str:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
        elif "mistral" in self.args.hf_model_str:
            model_inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded[0].split("[/INST]")[-1].strip()
        elif "vicuna" in self.args.hf_model_str or "Wizard" in self.args.hf_model_str:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        raise ValueError
