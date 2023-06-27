import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import json
import os

# URLs to scrape from
urls = [
    "https://scikit-learn.org/stable/faq.html#what-is-the-project-name-a-lot-of-people-get-it-wrong",
    "https://seaborn.pydata.org/faq.html",
    "https://pytorch.org/docs/stable/notes/faq.html",
    "https://docs.scrapy.org/en/latest/faq.html",
    "https://lightgbm.readthedocs.io/en/latest/FAQ.html",
    "https://doc.pypy.org/en/latest/faq.html",
    "https://docs.python.org/3/faq/general.html",
    "https://docs.python.org/3/faq/programming.html#why-am-i-getting-an-unboundlocalerror-when-the-variable-has-a-value",
    "https://docs.python.org/3/faq/extending.html",
]

# Names of the libraries
names = [
    'sklearn',
    'seaborn',
    'pytorch',
    'scrapy',
    'lightgbm',
    'pypy',
    'python general',
    'python programming',
    'python extending',
]

# Target tags to scrape
targets = ['h3', 'h3', 'h2', 'h2', 'h3', 'h2', 'h3', 'h3', 'h2']

# Create a list of dictionaries
pages = [{'name': names[i], 'url': urls[i], 'target': targets[i]} for i in range(len(names))]

# Get content between tags as markdown
def find_content_between(x):
    res = ""
    while True:
        x = x.next_sibling

        if not x or x.name == 'h3':
            break
        res += str(x)

    return md(res)


if __name__ == "__main__":
    # Loop over the pages and scrape
    for x in pages:
        page = requests.get(x['url'])
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find_all(x['target'])

        questions = []
        for i in results:
            questions.append({'instruction': i.text, 'context': x['name'], 'output': find_content_between(i)})

        exclusion = ['Table of Contents', 'This Page', 'Navigation']
        questions = [i for i in questions if i['instruction'] not in exclusion]
        
        # Write to output file
        with open('output.jsonl', 'a') as outfile:
            for entry in questions:
                json.dump(entry, outfile)
                outfile.write('\n')




