import json
import logging
from overrides import overrides
from spacy.tokens.token import Token
from typing import Any, Dict, List, Tuple

import my_util

from allennlp.models import Model

from allennlp.common.util import JsonDict
from allennlp.common.file_utils import cached_path

from allennlp.data.instance import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp.predictors.predictor import Predictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register("bidafplus_predictor")
class CoQAPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language="en_core_web_sm")

    def predict(self, jsonline: str) -> JsonDict:
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects json that looks like the original data file.
        """
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)

        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]

        logger.info("Reading the dataset...")

        paragraph_json = json_dict

        # for paragraph_json in dataset:
        paragraph = paragraph_json["story"]
        tokenized_paragraph = self._tokenizer.split_words(paragraph)
        questions = paragraph_json["questions"]
        golden_answers = paragraph_json["answers"]
        self.handle_unknown_answers(golden_answers, len(paragraph))
        metadata = {}
        paragraph_id = paragraph_json["id"]
        metadata["instance_id"] = [str(paragraph_id) + "_" + str(ques["turn_id"]) for ques in questions]

        if (len(metadata["instance_id"]) > 15):
            metadata["instance_id"] = metadata["instance_id"][:15]

        question_text_list = [ques["input_text"].strip().replace("\n", "") for ques in questions]
        if (len(question_text_list) > 15):
            question_text_list = question_text_list[:15]

        answer_texts_list = [[answer["span_text"]] for answer in golden_answers]
        if (len(answer_texts_list) > 15):
            answer_texts_list = answer_texts_list[:15]

        metadata["question"] = question_text_list
        metadata["answer_texts_list"] = answer_texts_list

        span_start_list = [[answer["span_start"]] for answer in golden_answers]
        span_end_list = [[answer["span_end"]] for answer in golden_answers]
        if (len(span_end_list) > 15):
            span_end_list = span_end_list[:15]

        # for st_list, an_list in zip(span_starts_list, answer_texts_list):
        #     span_ends = [start + len(answer) for start, answer in zip(st_list, an_list)]
        #     span_ends_list.append(span_ends)

        yesno_list = [str("x") for ques in questions][:15]
        followup_list = [str("n") for ques in questions][:15]
        instance = self._dataset_reader.text_to_instance(question_text_list,
                                                         paragraph,
                                                         span_start_list,
                                                         span_end_list,
                                                         tokenized_paragraph,
                                                         yesno_list,
                                                         followup_list,
                                                         metadata)
        return instance

    def text_to_instance(self,  # type: ignore
                         question_text_list: List[str],
                         passage_text: str,
                         start_span_list: List[List[int]] = None,
                         end_span_list: List[List[int]] = None,
                         passage_tokens: List[Token] = None,
                         yesno_list: List[int] = None,
                         followup_list: List[int] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we"ll actually use for supervision.
        answer_token_span_list = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for start_list, end_list in zip(start_span_list, end_span_list):
            token_spans: List[Tuple[int, int]] = []
            for char_span_start, char_span_end in zip(start_list, end_list):
                (span_start, span_end), error = my_util.char_span_to_token_span(passage_offsets,
                                                                             (char_span_start, char_span_end))
                if error:
                    logger.debug("Passage: %s", passage_text)
                    logger.debug("Passage tokens: %s", passage_tokens)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                    logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
                token_spans.append((span_start, span_end))
            answer_token_span_list.append(token_spans)
        question_list_tokens = [self._tokenizer.tokenize(q) for q in question_text_list]
        # Map answer texts to "CANNOTANSWER" if more than half of them marked as so.
        additional_metadata["answer_texts_list"] = [util.handle_cannot(ans_list) for ans_list \
                                                    in additional_metadata["answer_texts_list"]]
        return util.make_reading_comprehension_instance_quac(question_list_tokens,
                                                             passage_tokens,
                                                             self._token_indexers,
                                                             passage_text,
                                                             answer_token_span_list,
                                                             yesno_list,
                                                             followup_list,
                                                             additional_metadata,
                                                             self._num_context_answers)

    def handle_unknown_answers(self, answers, plen):
        for ans in answers:
            if ans["span_start"] < 0:
                ans["span_start"] = 0
            if ans["span_end"] < 0:
                ans["span_end"] = plen - 1
