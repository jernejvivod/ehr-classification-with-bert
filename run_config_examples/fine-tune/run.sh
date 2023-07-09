#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

data_file_path=$(find "$script_path/data/" -name '*train.txt' | head -n 1)

if [[ -z "$data_file_path" ]]; then
  echo "Error: no train data file found in data directory."
  exit 1
fi

# fine-tune
fine_tune_command="python3 $script_path/../../ehr_classification_with_bert \
          fine-tune \
          --data-file-path $data_file_path \
          --model-save-path $output_dir "

if [[ -v model_type ]]; then
  fine_tune_command+="--model-type $model_type "
fi

if [[ -v emb_model_path ]]; then
  fine_tune_command+="--emb-model-path $emb_model_path "
fi

if [[ -v hidden_size ]]; then
  fine_tune_command+="--hidden-size $hidden_size "
fi

if [[ -v n_labels ]]; then
  fine_tune_command+="--n-labels $n_labels "
fi

if [[ -v base_bert_model ]]; then
  fine_tune_command+="--base-bert-model $base_bert_model "
fi

if [[ -v freeze_emb_model && "$freeze_emb_model" == "true" ]]; then
  fine_tune_command+="--freeze-emb-model "
fi

if [[ -v n_epochs ]]; then
  fine_tune_command+="--n-epochs $n_epochs "
fi

if [[ -v batch_size ]]; then
  fine_tune_command+="--batch-size $batch_size "
fi

if [[ -v truncate_dataset_to ]]; then
  fine_tune_command+="--truncate-dataset-to $truncate_dataset_to "
fi

if [[ -v split_long_examples && "$split_long_examples" == "true" ]]; then
  fine_tune_command+="--split-long-examples "
fi

if [[ -v emb_model_method ]]; then
  fine_tune_command+="--emb-model-method $emb_model_method "
fi

if [[ -v emb_model_args ]]; then
  fine_tune_command+="--emb-model-args $emb_model_args "
fi

eval "$fine_tune_command"
