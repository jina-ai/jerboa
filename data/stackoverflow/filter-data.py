import json
import os
import random
import xml.etree.ElementTree as ET

base_path = '/path/to/SO/dump'
filepath = os.path.join(base_path, 'Posts.xml')
filtered_path = os.path.join(base_path, 'so-filtered.json')

# maps from post `Id` to another dictionary containing all attributes
# as well as a `answer` key
# `answer` is yet another dictionary containing the top answer
# filters:
# >= 10 upvotes for Q and A, answer length between 1200 and 4096 chars, not written in first person, no linking to other answers
data_filtered = dict()


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix) :]


def satisfies_question_filters(elem):
    return int(elem.attrib['Score']) >= 10


def satisfies_answer_filters(elem):
    if not int(elem.attrib['Score']) >= 10:
        return False

    answer_text = elem.attrib['Body']
    if len(answer_text) > 4096 or len(answer_text) < 1200:
        return False

    return (
        ' I ' not in answer_text
        and ' me ' not in answer_text
        and 'as mentioned' not in answer_text
    )


def filter():
    """For filtering the raw SO data dump"""
    for i, (event, elem) in enumerate(ET.iterparse(filepath, events=("start", "end"))):
        if elem.tag == 'row':
            if elem.attrib['PostTypeId'] == '1' and satisfies_question_filters(elem):
                post_id = elem.attrib['Id']
                # entry could already exist if answer comes before the question in the dataset
                existing_entry = data_filtered.get('post_id', None)
                if existing_entry is not None:
                    existing_answer = existing_entry.get('Answer', None)
                else:
                    existing_answer = None

                data_filtered[post_id] = elem.attrib
                data_filtered[post_id]['Answer'] = existing_answer

            if elem.attrib['PostTypeId'] == '2' and satisfies_answer_filters(elem):
                parent_id = elem.attrib['ParentId']
                # handle care where parent (question) entry does not already exist
                if data_filtered.get(parent_id, None) is None:
                    data_filtered[parent_id] = dict()
                # check if this is best answer
                existing_answer = data_filtered[parent_id].get('Answer', None)
                if existing_answer is None or int(elem.attrib['Score']) > int(
                    existing_answer['Score']
                ):
                    data_filtered[parent_id]['Answer'] = elem.attrib

            if event == 'end':
                elem.clear()

        if i % 100000 == 0:
            print(
                f'Iterated through {i / 1_000_000}M posts, collected {len(data_filtered)} filtered datapoints'
            )

    print('------------------------')
    print('Filtering complete')
    print(f'Iterated through {i} posts (questions + answers)')
    print(f'Collected {len(data_filtered)} filtered datapoints')
    with open(filtered_path, 'w') as fp:
        json.dump(data_filtered, fp)


def get_all_tags(path_to_filtered):
    """Creates a set of all SO tags present in the dataset"""
    with open(path_to_filtered, 'r') as fp:
        data = json.load(fp)
    tags = set()
    for i, (key, value) in enumerate(data.items()):
        tags.update(value.get('Tags', '').split('><'))
        if i % 100_000 == 0:
            print(
                f'Iterated through {i / 1_000_000}M posts, collected {len(tags)} tags'
            )
    return tags


def get_k_answers(path_to_filtered, k):
    with open(path_to_filtered, 'r') as fp:
        data = json.load(fp)
    answers = []
    for i, (key, value) in enumerate(data.items()):
        answer = value.get('Answer', None)
        if answer is not None:
            answers.append(answer['Body'])
        if len(answers) >= k:
            break
    return answers


def html_to_markdown(path_to_filtered, out_path):
    """Converts data from html to markdown, including code snippets"""
    from markdownify import markdownify as md

    def code_language_callback(el):
        # get the language of a given code block from html class
        return (
            remove_prefix(el['class'][0], prefix='lang-')
            if el.has_attr('class')
            else None
        )

    with open(path_to_filtered, 'r') as fp:
        data = json.load(fp)
    posts_mkdown = dict()
    for i, (key, value) in enumerate(data.items()):
        posts_mkdown[key] = value

        question_html = value.get('Body', None)
        if question_html is None:
            # no question body to convert, skip
            # not deleting the entry because the question can still have a title
            continue
        question_mkdown = md(
            question_html, code_language_callback=code_language_callback
        )
        posts_mkdown[key]['Body'] = question_mkdown

        answer_html = value.get('Answer', None)
        if answer_html is None:
            # has no answer, delete entry
            del posts_mkdown[key]
        else:
            answer_mkdown = md(
                answer_html['Body'], code_language_callback=code_language_callback
            )
            posts_mkdown[key]['Answer']['Body'] = answer_mkdown
        if i % 10_000 == 0:
            print(f'Converted {i} posts')

    print(f'Saving result to {out_path}')
    with open(out_path, 'w') as fp:
        json.dump(posts_mkdown, fp)


def filter_by_score(path_to_data, min_score):
    """Remove datapoints that don't have at least min_score"""
    with open(path_to_data, 'r') as fp:
        data = json.load(fp)
    filtered = dict()
    for i, (key, value) in enumerate(data.items()):
        score = int(value['Score']) if 'Score' in value else 0
        if not score:
            continue
        if score >= min_score:
            filtered[key] = value

    return filtered


def to_lima_format(path_to_data, out_path):
    """Converts from SO json to LIMA format (.jsonl)"""
    with open(path_to_data, 'r') as fp:
        data = json.load(fp)

    with open(out_path, 'w') as outfile:
        for i, (key, value) in enumerate(data.items()):
            score = int(value['Score']) if 'Score' in value else 0
            if not score:
                continue

            datapoint = dict()
            # sample between title and body of the question, following LIMA paper
            question = random.sample((value['Body'], value['Title']), 1)
            datapoint['conversations'] = question + [value['Answer']['Body']]
            datapoint['source'] = 'stackoverflow'

            # write to file
            json.dump(datapoint, outfile)
            outfile.write('\n')


def sample_from_jsonl(path_to_data, out_path, num_samples):
    """Takes a dataset in jsonl format and samples num_samples data points"""
    # read
    with open(path_to_data, 'r') as file:
        jsonl_list = [line.rstrip() for line in file]
    # samples
    sampled_jsonl_list = random.sample(jsonl_list, num_samples)
    # write
    with open(out_path, 'w') as outfile:
        for line in sampled_jsonl_list:
            json.dump(line, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    ...
