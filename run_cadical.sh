#!/bin/bash

# Configuration
INPUT_DIR="sat2017"
OUTPUT_DIR="runs"
CADICAL_SCRIPT="cadical.sh"
CADICAL_WORKDIR="$HOME/source/cadical-simd"

# CaDiCaL parameters
CADICAL_TIMEOUT=3600  # in seconds (1 hour, matching walltime in cadical.sh)
CADICAL_OPTIONS="-n"

# PBS parameters
PBS_QUEUE="zeus_new_q"
PBS_WALLTIME="01:05:00"

# Function to generate a timestamp
get_timestamp() {
    date "+%Y%m%d_%H%M%S"
}

# Function to create directory structure
create_directory_structure() {
    local run_id=$1
    mkdir -p "$OUTPUT_DIR/$run_id"/{results,stdout,stderr}
}

# Function to get base filename
get_base_filename() {
    basename "$1" .cnf
}

# Function to prepare output filenames
prepare_output_files() {
    local run_id=$1
    local base_name=$2
    echo "$OUTPUT_DIR/$run_id/results/${base_name}_result.txt" \
         "$OUTPUT_DIR/$run_id/stdout/${base_name}_stdout.txt" \
         "$OUTPUT_DIR/$run_id/stderr/${base_name}_stderr.txt"
}

# Function to submit job
submit_job() {
    local input_file=$1
    local base_name=$2
    local result_file=$3
    local stdout_file=$4
    local stderr_file=$5

    qsub \
        -N "cadical_${base_name}" \
        -q "$PBS_QUEUE" \
        -l walltime=$PBS_WALLTIME \
        -v arg="$input_file $CADICAL_OPTIONS -t $CADICAL_TIMEOUT",out="$result_file",PBS_O_WORKDIR="$CADICAL_WORKDIR" \
        -o "$stdout_file" \
        -e "$stderr_file" \
        "$CADICAL_SCRIPT"
}

# Main execution
main() {
    local run_id=$(get_timestamp)
    create_directory_structure "$run_id"

    for input_file in "$INPUT_DIR"/*.cnf; do
        local base_name=$(get_base_filename "$input_file")
        read -r result_file stdout_file stderr_file < <(prepare_output_files "$run_id" "$base_name")
        submit_job "$input_file" "$base_name" "$result_file" "$stdout_file" "$stderr_file"
    done

    echo "Jobs submitted. Results will be in: $OUTPUT_DIR/$run_id/"
}

# Run the main function
main