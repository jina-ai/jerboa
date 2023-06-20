"""
A dedicated helper to manage templates and prompt building.
"""

import inspect
import json
import os.path as osp
from typing import Union

import jerboa


class Prompter(object):
    __slots__ = ("template_name", "template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self.template_name = template_name
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"

        jerboa_path = osp.dirname(inspect.getfile(jerboa))
        file_name = osp.join("templates", f"{template_name}.json")
        file_name = osp.join(jerboa_path, file_name)
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if self.template_name == "lima":
            res = self.template["prompt"].format(conversation=instruction)
        else:
            if input:
                res = self.template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
