import os
import json
import logging
from typing import Any, Dict, List, Tuple
from spacy.tokens.token import Token
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("coqa_seq2seq_reader")
class CoqaSeq2SeqReader(DatasetReader):
    """
        Read the coqa dataset and create a dataset suitable for a
        ``SimpleSeq2Seq`` model, or any model with a matching API.

        Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

        The output of ``read`` is a list of ``Instance`` s with the fields:
            source_tokens: ``TextField`` and
            target_tokens: ``TextField``

        `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

        Parameters
        ----------
        source_tokenizer : ``Tokenizer``, optional
            Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
            to ``WordTokenizer()``.
        target_tokenizer : ``Tokenizer``, optional
            Tokenizer to use to split the output sequences (during training) into words or other kinds
            of tokens. Defaults to ``source_tokenizer``.
        source_token_indexers : ``Dict[str, TokenIndexer]``, optional
            Indexers used to define input (source side) token representations. Defaults to
            ``{"tokens": SingleIdTokenIndexer()}``.
        target_token_indexers : ``Dict[str, TokenIndexer]``, optional
            Indexers used to define output (target side) token representations. Defaults to
            ``source_token_indexers``.
        source_add_start_token : bool, (optional, default=True)
            Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
        """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.question_tag = '<q>'

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        # print("file_path = ", file_path)
        logger.info("Reading file at %s", file_path)
        
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']

        # READ THE BIDAF++ OUTPUTS
        bidafplus_output_filename = os.path.join(os.path.dirname(os.path.realpath(file_path)),
                                                                'bidafplus_output_formatted.json')
        with open(bidafplus_output_filename) as bidafplus_outputs:
            best_span_str_json = json.load(bidafplus_outputs)
            best_span_str = best_span_str_json['data']
        
        logger.info("Reading the dataset")
        for paragraph_json in dataset:
            all_questions = paragraph_json['questions']
            golden_answers = paragraph_json['answers']
            paragraph_id = paragraph_json['id']

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
                yield self.text_to_instance(ques_rat_list[i], answers_list[i], metadata)
                # yield self.text_to_instance(rationale_list[i], answers_list[i])

    @overrides
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
            return Instance({"source_tokens": source_field, 
                             "target_tokens": target_field}) 
        else:
            return Instance({"source_tokens": source_field}) 
