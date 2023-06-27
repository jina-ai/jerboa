import json
import os
import xml.etree.ElementTree as ET

base_path = '/path/to/stackoverflow/dump'
filepath = os.path.join(base_path, 'Posts.xml')
output_path = os.path.join(base_path, 'so-filtered.json')

# maps from post `Id` to another dictionary containing all attributes
# as well as a `answer` key
# `answer` is yet another dictionary containing the top answer
# filters:
# >= 10 upvotes for Q and A, answer length between 1200 and 4096 chars, not written in first person, no linking to other answers
data_filtered = dict()


def satisfies_question_filters(elem):
    return int(elem.attrib['Score']) >= 10


def satisfies_answer_filters(elem):
    if not int(elem.attrib['Score']) >= 10:
        return False

    answer_text = elem.attrib['Body']
    if not 1200 < len(answer_text) < 4096:
        return False

    exclusion_words = [' I ', ' me ', 'as mentioned']
    return not any(word in answer_text for word in exclusion_words)
    
for i, (event, elem) in enumerate(ET.iterparse(filepath, events=("start", "end"))):
    if elem.tag == 'row':
        if elem.attrib['PostTypeId'] == '1' and satisfies_question_filters(elem):
            # Question
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
            # Answer
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
with open(output_path, 'w') as fp:
    json.dump(data_filtered, fp)
