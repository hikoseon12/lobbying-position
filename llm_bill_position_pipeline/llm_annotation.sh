#!/bin/bash

INPUT_PATH=${1:-"llm_input/llm_bill_position_input_df.csv"}
OUTPUT_DIR=${2:-"llm_output"}
OUTPUT_NAME=${3:-"llm_bill_position_prediction_df.csv"}
MODEL=${4:-"gpt-4-1106-Preview"}
API_VERSION=${5:-"2023-09-01-preview"}
OPENAI_API_KEY=${6:-"null"}
AZURE_ENDPOINT=${7:-"null"}
NUM_CORES=${8:-"1"}


python llm_annotation.py \
    --input_path $INPUT_PATH \
    --output_dir $OUTPUT_DIR \
    --output_name $OUTPUT_NAME \
    --model $MODEL \
    --api_version $API_VERSION \
    --api_key $OPENAI_API_KEY \
    --azure_endpoint $AZURE_ENDPOINT \
    --num_cores $NUM_CORES 
