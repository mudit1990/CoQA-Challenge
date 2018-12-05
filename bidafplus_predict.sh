#!/bin/sh

allennlp predict \
	--output-file out \
	--include-package bidafplus_model \
	--include-package bidafplus_coqa_reader \
	--include-package bidafplus_predictor \
	--predictor bidafplus_predictor \
	--use-dataset-reader \
	models/bidafplus_model.tar.gz data/dev/coqa-dev-v1.0.json
