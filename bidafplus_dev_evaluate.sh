#!/bin/sh

allennlp evaluate \
	--output-file bidafplus_dev_eval.txt \
	--cuda-device 0 \
	--include-package bidafplus_model \
	--include-package bidafplus_coqa_reader \
	models/custom_data_model.tar.gz data/dev/coqa-dev-v1.0.json
