The data here is obtained from the StackOverflow data dump, [here](https://archive.org/details/stackexchange).

Some filtering is obtained according to the scrip in `filter-data.py`.
This is not (yet) intended to be the final dataset, just a manageable set of candidate QA pairs.

This set of QA pairs can be found in an S3 bucket [here](https://stackoverflow-filtered.s3.eu-central-1.amazonaws.com/v1-rough-filtering.json).
Every entry in the json is a question with a bunch of associated metadata.
Most important are the `Body` and the `Answer`.

There are more filtered datasets available as well:
- [mkdown-score-20.json](https://s3.console.aws.amazon.com/s3/object/stackoverflow-filtered?region=eu-central-1&prefix=mkdown-score-20.json) contains all QA pairs that have at least a score of 20. It is in the same format as the dataset above. Qs and As have been converted from HTML to Markdown.
- [mkdown-score-20.jsonl](https://s3.console.aws.amazon.com/s3/object/stackoverflow-filtered?region=eu-central-1&prefix=mkdown-score-20.jsonl) is the same dataset as `mkdown-score-20.json`, but in LIMA format in a `.jsonl` file. This means that it does not contain metadata (such as scores).
- [mkdown-score-20-sample-1000.jsonl](https://s3.console.aws.amazon.com/s3/object/stackoverflow-filtered?region=eu-central-1&prefix=mkdown-score-20-sample-1000.jsonl) is a dataset of 1000 samples from `mkdown-score-20.jsonl`.