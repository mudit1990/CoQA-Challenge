#!/bin/sh

allennlp train elmo_seq2seq_context.json -s ../output_s2s_deep_enc_elmo --include-package seq2seq_model --include-package coqa_seq2seq_reader
