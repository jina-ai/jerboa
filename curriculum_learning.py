from typing import List, Dict, Tuple, Any

from langchain.schema import BaseOutputParser, OutputParserException
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from jerboa.utils.prompter import Prompter


class NumericOutputParser(BaseOutputParser):
    range_values: Tuple

    @property
    def _valid_values(self) -> List[int]:
        start, end = self.range_values
        return [value for value in range(start, end+1)]

    def parse(self, response: str) -> Any:
        try:
            response = response.strip()
            if int(response) in self._valid_values:
                return int(response)
        except Exception as e:
            raise OutputParserException(
                f"Response '{response}' is not one of the "
                f"expected values: {self._valid_values}"
            )

    def get_format_instructions(self) -> str:
        return f"Only return an integer in the range: [{self.range_values[0]}, {self.range_values[1]}]"


parser = NumericOutputParser(range_values=(1, 10))


llm = OpenAI(temperature=0.9, model_name='gpt-3.5-turbo')
prompt = PromptTemplate(
    input_variables=["answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="rate the complexity of the following answer from 1 to 10 with 10 being the most complex answer:\n {answer}\n{format_instructions}",
)

chain = LLMChain(llm=llm, prompt=prompt)


def rate_answer(answer):
    res = chain.run(answer)
    return parser.parse(res)

def rate_with_retry(answer, retries=3):
    for _ in range(retries):
        try:
            return rate_answer(answer)
        except:
            pass
    return None

def compute_rate_field(element):
    # prompter = Prompter('alpaca')
    # element['rate'] = rate_with_retry(prompter.generate_prompt(element['instruction'], element['input'], element['output']))
    element['rate'] = rate_with_retry(element['output'])
    return element

def drop_invalid_rates(element):
    return element['rate'] is not None

def process_dataset(dataset):
    dataset = dataset.map(compute_rate_field, num_proc=4)
    dataset = dataset.filter(drop_invalid_rates)
    return dataset


if __name__ == '__main__':
    from datasets import load_dataset
    from datasets import disable_caching
    disable_caching()
    dataset = load_dataset('yahma/alpaca-cleaned')
    train_sample = dataset['train']

    train_sample = process_dataset(train_sample)
    train_sample.to_json('rated.json')
    import pandas as pd
    df = pd.DataFrame(train_sample)
    df.to_csv('rated.csv')
