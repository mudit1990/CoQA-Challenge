#!/bin/sh

allennlp predict \
	--output-file data/dev/bidafplus_output.json \
	--include-package bidafplus_model \
	--include-package bidafplus_coqa_reader \
	--include-package bidafplus_predictor \
	--cuda-device 0 \
	--predictor bidafplus_predictor \
	--use-dataset-reader \
	models/custom_data_model.tar.gz data/dev/coqa-dev-v1.0.json
