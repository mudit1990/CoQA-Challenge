import json
import logging
from overrides import overrides
from spacy.tokens.token import Token
from typing import Any, Dict, List, Tuple

import my_util

from allennlp.models import Model

from allennlp.common.util import JsonDict, sanitize
from allennlp.common.file_utils import cached_path

from allennlp.data.instance import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp.predictors.predictor import Predictor

@Predictor.register('coqa_seq2seq_predictor')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language="en_core_web_sm")

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        del outputs["logits"]
        del outputs["class_probabilities"]
        return sanitize(outputs)

    def predict(self, source: str) -> JsonDict:
        pred_json = self.predict_json({"source" : source})
        return pred_json

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        # print(json_dict)
        paragraph_json = json_dict
        all_questions = paragraph_json['questions']
        golden_answers = paragraph_json['answers']
        paragraph_id = paragraph_json['id']

        # READ THE BIDAF++ OUTPUTS
        bidafplus_output_filename = os.path.join(os.path.dirname(os.path.realpath(file_path)),
                                                                'bidafplus_output_formatted.json')
        with open(bidafplus_output_filename) as bidafplus_outputs:
            best_span_str_json = json.load(bidafplus_outputs)
            best_span_str = best_span_str_json['data']

        # extractive outputs from BIDAF++
        best_span_str_list = best_span_str[paragraph_id]
        
        # metadata
        metadata = {}
        metadata['paragraph_id'] = paragraph_id
        metadata['questions'] = [ques["input_text"].strip().replace("\n", "") for ques in all_questions][:15]

        questions_list = [ques["input_text"].strip().replace("\n", "") for ques in all_questions][:15]
        golden_rationale_list = [answer['span_text'].strip().replace("\n", "") for answer in golden_answers][:15]
        answers_list = [answer['input_text'].strip().replace("\n", "") for answer in golden_answers][:15]
        bidafplus_rationale_list = [answer['answer_text'].strip().replace("\n", "") for answer in best_span_str_list][:15]
        ques_rat_list = [' '.join([bidafplus_rationale_list[i], self.question_tag, questions_list[i]]) for i in
                         range(len(questions_list))]
        for i in range(len(questions_list)):
            yield self.text_to_instance(ques_rat_list[i], answers_list[i], paragraph_id, i)
            # yield self.text_to_instance(rationale_list[i], answers_list[i])

    def text_to_instance(self, source_string: str, 
                         target_string: str = None, 
                         paragraph_id: str = None, 
                         turn_id: int = 0) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._token_indexers)
        if target_string is not None:
            tokenized_target = self._tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._token_indexers)
            return Instance({"source_tokens": source_field})
        else:
            return Instance({"source_tokens": source_field})

