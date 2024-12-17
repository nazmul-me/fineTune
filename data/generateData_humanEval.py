from datasets import load_dataset
data = load_dataset("openai_humaneval")

print(data)

concate_all_data = [ c['prompt']+c['canonical_solution'] for c in data['test']]
concate_data = concate_all_data[:82]
print(concate_data[0])

import json
with open('code_segments_humaneval.json','w') as f:
    for row in concate_data:
        dic = {"content": str(row)}
        ob = json.dumps(dic)
        f.write(ob)
        f.write('\n')
f.close()