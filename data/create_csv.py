import pandas as pd
import json
import json

import json
with open('/root/jerboa/jerboa/eval_corrected.jsonl') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df.to_csv('.comparison.csv')