#!/bin/sh

allennlp evaluate \
	--output-file seq2seq_dev_eval.txt \
	--cuda-device 0 \
	--include-package seq2seq_model \
	--include-package coqa_seq2seq_reader \
	models/s2s_model.tar.gz data/dev/coqa-dev-v1.0.json
