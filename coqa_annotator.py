import json
import string
import pickle
import pprint
import os
from dateutil.parser import parse
from tqdm import tqdm


def annotate_coqa(file_path, annot_path):
    """
        Reads a JSON-formatted CoQA dataset and annotates each question-answer pair
        with span_overlap(double), ground-truth answer length, is yes/no(binary), has date (binary).
        The output of the code is a dictionary with key as id_turn_id & value as annotations.
        The annotations is a dictionary with the following structure
        {
            'span_overlap': double,
            'gt_ans_len': int,
            'is_yn': True/False
            'has_num_date': True/False
        }
        <id, turn_id, span_overlap, gt_answer_len, is_yn, has_date>
    """
    pp = pprint.PrettyPrinter(indent=2)
    print("Reading file a", file_path)
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    annot_data = {}
    for i in tqdm(range(len(dataset))):
        paragraph_json = dataset[i]
        golden_answers = paragraph_json['answers']
        paragraph_id = paragraph_json['id']
        for ans in golden_answers:
            if ans['turn_id'] > 15:
                break
            annot = {}
            instance_id = str(paragraph_id) + "_" + str(ans['turn_id'])
            gt_ans = ans['input_text']
            span = ans['span_text']
            annot['span_overlap'] = get_span_overlap(span, gt_ans)
            annot['gt_ans_len'] = len(normalize_answer(gt_ans).split())
            annot['is_yn'] = True if gt_ans.lower() == 'yes' or gt_ans.lower() == 'no' else False
            annot['has_num_date'] = contains_numbers_datetime(gt_ans)
            annot_data[instance_id] = annot

    # pp.pprint(annot_data)
    annot_file = os.path.join(annot_path, 'data_annotations.pkl')
    with open(annot_file, 'wb') as ap:
        pickle.dump(annot_data, ap)


def contains_numbers_datetime(gt_ans):
    ans_words = normalize_answer(gt_ans).split()
    for w in ans_words:
        if is_date(w) or is_number(w):
            return True
    return False


def is_number(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    # def remove_articles(text):
    #     return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def get_span_overlap(span, ans):
    span_words = set(normalize_answer(span).split())
    ans_words = normalize_answer(ans).split()
    hit = 0
    for w in ans_words:
        if w in span_words:
            hit += 1
    return hit / len(ans_words)


annotate_coqa('data/test/test.json', 'data/test')
