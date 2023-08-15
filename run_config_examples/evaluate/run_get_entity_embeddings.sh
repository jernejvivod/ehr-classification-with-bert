#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

train_data_file=$(find "$script_path/data/" -name '*train.txt' | head -n 1)
test_data_file=$(find "$script_path/data/" -name '*test.txt' | head -n 1)

if [[ -z "$train_data_file" ]]; then
  echo "Error: no train data file found in data directory."
  exit 1
fi

if [[ -z "$test_data_file" ]]; then
  echo "Error: no test data file found in data directory."
  exit 1
fi

# get embeddings
get_embeddings_command="python3 $script_path/../../classification_with_embeddings \
          get-entity-embeddings \
          --train-data-path $train_data_file \
          --output-dir $output_dir "

if [[ -v method ]]; then
  get_embeddings_command+="--method $method "

  if [[ "$method" == "word2vec" && -v word2vec_args ]]; then
    get_embeddings_command+="--word2vec-args \"${word2vec_args//,/ }\" "
  fi

  if [[ "$method" == "fasttext" && -v fasttext_args ]]; then
    get_embeddings_command+="--fasttext-args  \"${fasttext_args//,/ }\""
  fi

  if [[ "$method" == "starspace" && -v starspace_args ]]; then
    get_embeddings_command+="--starspace-args \"${starspace_args//,/ }\" "
  fi
fi

echo "$get_embeddings_command"
eval "$get_embeddings_command"
