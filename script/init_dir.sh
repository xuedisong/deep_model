#!/usr/bin/env bash
set -o errexit
script_dir=$(
  cd "$(dirname "$0")" || exit
  pwd
)
project_dir=$(dirname "$script_dir")
conf_dir="$project_dir/conf"
script_dir="$project_dir/script"
log_dir="$project_dir/log"
data_dir="$project_dir/data"
target_dir="$project_dir/target"
sql_dir="$project_dir/sql"
tool_dir="$project_dir/tool"
src_dir="$project_dir/src"
model_dir="$project_dir/model"

PROJECT_DIR=$(dirname "$script_dir")
CONF_DIR="$project_dir/conf"
SCRIPT_DIR="$project_dir/script"
LOG_DIR="$project_dir/log"
DATA_DIR="$project_dir/data"
TARGET_DIR="$project_dir/target"
SQL_DIR="$project_dir/sql"
TOOL_DIR="$project_dir/tool"
SRC_DIR="$project_dir/src"
MODEL_DIR="$project_dir/model"

echo "init_dir.sh conf_dir:$conf_dir"
