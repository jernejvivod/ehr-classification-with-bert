#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

data_file_path_p1=$(find "$script_path/data/" -name '*p1_train.txt' | sort | tr '\n' ' ')
data_file_path_p2=$(find "$script_path/data/" -name '*p2_train.txt' | sort | tr '\n' ' ')

val_file_path_p1=$(find "$script_path/data/" -name '*p1_val.txt' | sort | tr '\n' ' ')
val_file_path_p2=$(find "$script_path/data/" -name '*p2_val.txt' | sort | tr '\n' ' ')

bert_model_path=$(find "$script_path/data/" -name '*bert_model.pth' | sort | tr '\n' ' ')
emb_model_path=$(find "$script_path/data/" -name '*emb_model.pth' | sort | tr '\n' ' ')

if [[ -z "$data_file_path_p1" ]]; then
  echo "Error: no first segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$data_file_path_p2" ]]; then
  echo "Error: no second segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$n_labels" ]]; then
  echo "Error: number of unique labels should be specified in config file."
  exit 1
fi

# fine-tune
fine_tune_command="python3 $script_path/../../ehr_classification_with_bert \
          fine-tune \
          --data-file-path $data_file_path_p1 $data_file_path_p2 \
          --n-labels $n_labels \
          --model-save-path $output_dir "

if [[ -n "$val_file_path_p1" && -n "$val_file_path_p2" ]]; then
  fine_tune_command+="--val-file-path $val_file_path_p1 $val_file_path_p2"
fi

if [[ -n "$eval_every_steps" ]]; then
  fine_tune_command+="--eval-every-steps $eval_every_steps "
fi

if [[ -n "$model_type" ]]; then
  fine_tune_command+="--model-type $model_type "
fi

if [[ -n "$bert_model_path" ]]; then
  fine_tune_command+="--bert-model-path $bert_model_path "
fi

if [[ -n "$emb_model_path" ]]; then
  fine_tune_command+="--emb-model-path $emb_model_path "
fi

if [[ -n "$hidden_size" ]]; then
  fine_tune_command+="--hidden-size $hidden_size "
fi

if [[ -n "$base_bert_model" ]]; then
  fine_tune_command+="--base-bert-model $base_bert_model "
fi

if [[ -n "$freeze_bert_model" && "$freeze_bert_model" == "true" ]]; then
  fine_tune_command+="--freeze-bert-model "
fi

if [[ -n "$freeze_emb_model" && "$freeze_emb_model" == "true" ]]; then
  fine_tune_command+="--freeze-emb-model "
fi

if [[ -n "$n_epochs" ]]; then
  fine_tune_command+="--n-epochs $n_epochs "
fi

if [[ -n "$batch_size" ]]; then
  fine_tune_command+="--batch-size $batch_size "
fi

if [[ -n "$truncate_dataset_to" ]]; then
  fine_tune_command+="--truncate-dataset-to $truncate_dataset_to "
fi

if [[ -n "$split_long_examples" && "$split_long_examples" == "true" ]]; then
  fine_tune_command+="--split-long-examples "
fi

if [[ -n "$emb_model_method" ]]; then
  fine_tune_command+="--emb-model-method $emb_model_method "
fi

if [[ -n "$emb_model_args" ]]; then
  fine_tune_command+="--emb-model-args $emb_model_args "
fi

echo "$fine_tune_command"
eval "$fine_tune_command"
