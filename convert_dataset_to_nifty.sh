#! /bin/bash

input_dir='./sorted'
output_dir='./sorted_nii'

# Assuming that the python script is placed under the same directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PYTHON=python3

${PYTHON} ${SCRIPT_DIR}/to_nifty.py  ${input_dir}/train_images  ${output_dir}/train_images
${PYTHON} ${SCRIPT_DIR}/to_nifty.py  ${input_dir}/train_labels  ${output_dir}/train_labels
${PYTHON} ${SCRIPT_DIR}/to_nifty.py  ${input_dir}/val_images    ${output_dir}/val_images
${PYTHON} ${SCRIPT_DIR}/to_nifty.py  ${input_dir}/val_labels    ${output_dir}/val_labels

