#!/bin/sh

allennlp predict \
	--output-file data/dev/s2s_output.json \
	--include-package seq2seq_model \
	--include-package coqa_seq2seq_reader \
	--include-package coqa_seq2seq_predictor \
	--predictor coqa_seq2seq_predictor \
	--use-dataset-reader \
	--cuda-device 0 \
	models/s2s_model.tar.gz data/dev/coqa-dev-v1.0.json
