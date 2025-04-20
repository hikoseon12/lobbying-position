#!/bin/bash

REPORT_DF_PATH=${1:-"llm_input/report_info_df.csv"}
OUTPUT_DIR=${2:-"llm_output"}
OUTPUT_NAME=${3:-"llm_bill_position_prediction_df.csv"}
FINAL_OUTPUT_NAME=${4:-"llm_bill_position_final_df.csv"}

python postprocess_llm_annotation.py \
    --report_df_path $REPORT_DF_PATH \
    --output_dir $OUTPUT_DIR \
    --output_name $OUTPUT_NAME \
    --final_output_name $FINAL_OUTPUT_NAME
