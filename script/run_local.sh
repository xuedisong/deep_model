#!/usr/bin/env bash
set -e
DIR_NAME=$(
  cd "$(dirname "$0")" || exit
  pwd
)
source "${DIR_NAME}"/init_dir.sh

dt="2022-01-01"
version="version_base"
#data_path=$data_dir/$dt/$version
model_path=$model_dir

function get_config() {
  model_data_dir="data_demo"
  data_path=$DATA_DIR/$model_data_dir

  train_data_file="$data_path/train_data.txt"
  test_data_file="$data_path/eval_data.txt"
  feature_info_file="$data_path/feature_info.txt"
  dict_file="$data_path/dict.txt"
  model_dir="$MODEL_DIR/$model_data_dir/$model_version"
  base_config="--feature_info=\${feature_info_file} --dict_path=\${dict_file} --model_dir=\${model_dir} --train_data=\${train_data_file} --test_data=\${test_data_file}"

  model_params_file="$CONF_DIR/model_params.sh"
  source "$model_params_file"
  model_config=$(awk -F "=" '{printf " --"$1"=$"$1}' "$model_params_file")

  eval "echo $base_config $model_config"
}

function run() {
  work_mode=$1
  config=$(get_config)
  echo "APP config:--work_mode="$work_mode" $config"
  python "$SRC_DIR"/tf_app.py --work_mode="$work_mode" $config
}

work_mode="train_and_eval"
run "$work_mode"
