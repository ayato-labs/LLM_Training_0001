with open(r'C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\LLM_Training\data\corpus.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        import json
        d = json.loads(line)
        print(f'Keys: {list(d.keys())}')
        text = d.get("text", "")
        print(f'Text sample: {text[:200]}')
        print()