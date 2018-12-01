#!/bin/sh

allennlp train bidafplus_context.json -s ../output --include-package bidafplus_model --include-package bidafplus_coqa_reader
