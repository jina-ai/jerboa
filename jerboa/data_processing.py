import multiprocessing as mp
from collections import defaultdict
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


def load_train_val_data(
    data_path: str,
    data_files: Optional[str],
    dataset_preprocessor: str,
    val_set_size: int,
    n_samples: int,
    debug: bool,
):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path, data_files=data_files)

    preprocessor = (
        PREPROCESSORS.get(dataset_preprocessor, None)
        or PREPROCESSORS_MAP[(data_path, data_files)]
    )
    data = preprocessor(data)

    if debug:
        train_data = data["train"].select(range(10))
        val_data = None
    elif val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle()
        val_data = train_val["test"].shuffle()
    else:
        train_data = data["train"].shuffle()
        val_data = None

    if n_samples:
        train_data = train_data.select(range(n_samples))

    return train_data, val_data


def process_element_redpajamas_ni_to_alpaca_format(element):
    output_dict = {}
    if len(element['definition']) > 0 and len(element['targets']) > 0:
        output_dict['instruction'] = element['definition'][0]
        output_dict['input'] = element['inputs']
        output_dict['output'] = element['targets'][0]
    return output_dict


def redpajamas_ni_to_alpaca_format(dataset: DatasetDict) -> DatasetDict:
    with mp.Pool(8) as pool:
        output_list = list(
            pool.imap(
                process_element_redpajamas_ni_to_alpaca_format,
                tqdm(dataset['train']),
                chunksize=5000,
            )
        )
    return DatasetDict({'train': Dataset.from_list(output_list)})


def process_element_redpajamas_p3_to_alpaca_format(element):
    output_dict = {
        'instruction': element['inputs'],
        'input': '',
        'output': element['targets'],
    }
    return output_dict


def process_element_lima_to_alpaca_format(element):
    output_dict = {
        'instruction': element['conversations'][0],
        'input': '',
        'output': element['conversations'][1],
    }
    return output_dict


def process_element_dolly_to_alpaca_format(element):
    output_dict = {
        'instruction': element['instruction'],
        'input': element['context'],
        'output': element['response'],
    }
    return output_dict


def process_element_dolly_hhrlhf_to_alpaca_format(element):
    output_dict = {
        'instruction': element['prompt'],
        'input': '',
        'output': element['response'],
    }
    return output_dict


def redpajamas_p3_to_alpaca_format(dataset: DatasetDict) -> DatasetDict:
    with mp.Pool(8) as pool:
        output_list = list(
            pool.imap(
                process_element_redpajamas_p3_to_alpaca_format,
                tqdm(dataset['train']),
                chunksize=5000,
            )
        )
    return DatasetDict({'train': Dataset.from_list(output_list)})


def lima_to_alpaca_format(dataset: DatasetDict) -> DatasetDict:
    with mp.Pool(8) as pool:
        output_list = list(
            pool.imap(
                process_element_lima_to_alpaca_format,
                tqdm(dataset['train']),
                chunksize=5000,
            )
        )
    return DatasetDict({'train': Dataset.from_list(output_list)})


def dolly_to_alpaca_format(dataset: DatasetDict) -> DatasetDict:
    with mp.Pool(8) as pool:
        output_list = list(
            pool.imap(
                process_element_dolly_to_alpaca_format,
                tqdm(dataset['train']),
                chunksize=5000,
            )
        )
    return DatasetDict({'train': Dataset.from_list(output_list)})


def dolly_hhrlhf_to_alpaca_format(dataset: DatasetDict) -> DatasetDict:
    with mp.Pool(8) as pool:
        output_list = list(
            pool.imap(
                process_element_dolly_hhrlhf_to_alpaca_format,
                tqdm(dataset['train']),
                chunksize=5000,
            )
        )
    return DatasetDict({'train': Dataset.from_list(output_list)})


PREPROCESSORS = {
    'redpajamas_ni_to_alpaca_format': redpajamas_ni_to_alpaca_format,
    'redpajamas_p3_to_alpaca_format': redpajamas_p3_to_alpaca_format,
    'lima_to_alpaca_format': lima_to_alpaca_format,
    'dolly_to_alpaca_format': dolly_to_alpaca_format,
    'default': None,
}
PREPROCESSORS_MAP = defaultdict(lambda: lambda x: x)
PREPROCESSORS_MAP[
    ('togethercomputer/RedPajama-Data-Instruct', "data/NI_decontaminated.jsonl.zst")
] = redpajamas_ni_to_alpaca_format
PREPROCESSORS_MAP[
    ('togethercomputer/RedPajama-Data-Instruct', "data/P3_decontaminated.jsonl.zst")
] = redpajamas_p3_to_alpaca_format
PREPROCESSORS_MAP[('databricks/databricks-dolly-15k', None)] = dolly_to_alpaca_format
PREPROCESSORS_MAP[('GAIR/lima', None)] = lima_to_alpaca_format
