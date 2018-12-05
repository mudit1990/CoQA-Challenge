import json

# Dev predictions
with open('dev/bidafplus_output.json') as f:
    lines = f.readlines()
    L = []
    for line in lines:
        passage_json = {}
        line_json = json.loads(line)
        _id, _ = line_json['qid'][0].split('_')
        passage_json['id'] = _id
        answers = line_json['best_span_str']
        qids = line_json['qid']
        passage_json['bidafplus_answers'] = []
        for turn, answer in zip(qids, answers):
            turn_id = int(turn.split('_')[1])
            tmp = {}
            tmp['turn_id'] = turn_id
            tmp['answer_text'] = answer
            passage_json['bidafplus_answers'].append(tmp)
        L.append(passage_json)
    out = {}
    out['data'] = L

    with open('dev/bidafplus_output_formatted.json', 'w') as f2:
        json.dump(out, f2, indent=2, separators=(',', ': '))

with open('test/bidafplus_output.json') as f:
    lines = f.readlines()
    L = []
    for line in lines:
        passage_json = {}
        line_json = json.loads(line)
        _id, _ = line_json['qid'][0].split('_')
        passage_json['id'] = _id
        answers = line_json['best_span_str']
        qids = line_json['qid']
        passage_json['bidafplus_answers'] = []
        for turn, answer in zip(qids, answers):
            turn_id = int(turn.split('_')[1])
            tmp = {}
            tmp['turn_id'] = turn_id
            tmp['answer_text'] = answer
            passage_json['bidafplus_answers'].append(tmp)
        L.append(passage_json)
    out = {}
    out['data'] = L

    with open('test/bidafplus_output_formatted.json', 'w') as f2:
        json.dump(out, f2, indent=2, separators=(',', ': '))
