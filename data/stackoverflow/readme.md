The data here is obtained from the StackOverflow data dump, [here](https://archive.org/details/stackexchange).

Some filtering is obtained according to the scrip in `filter-data.py`.
This is not (yet) intended to be the final dataset, just a manageable set of candidate QA pairs.

This set of QA pairs can be found in an S3 bucket [here](https://stackoverflow-filtered.s3.eu-central-1.amazonaws.com/v1-rough-filtering.json).
Every entry in the json is a question with a bunch of associated metadata.
Most important are the `Body` and the `Answer`.