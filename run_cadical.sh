#!/bin/bash

# Configuration
INPUT_DIR="sat2017"
OUTPUT_DIR="runs"
CADICAL_SCRIPT="cadical.sh"
CADICAL_WORKDIR="$HOME/source/cadical-simd"

# CaDiCaL parameters
CADICAL_TIMEOUT=3600  # in seconds
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
    local run_dir=$1
    mkdir -p "$run_dir"/{results,stdout,stderr}
}

# Function to get base filename
get_base_filename() {
    basename "$1" .cnf
}

# Function to prepare output filenames
prepare_output_files() {
    local run_dir=$1
    local base_name=$2
    echo "$run_dir/results/${base_name}_result.txt" \
         "$run_dir/stdout/${base_name}_stdout.txt" \
         "$run_dir/stderr/${base_name}_stderr.txt"
}

# Function to submit job and check success
submit_job() {
    local input_file=$1
    local base_name=$2
    local result_file=$3
    local stdout_file=$4
    local stderr_file=$5
    local failed_jobs_file=$6

    # Capture qsub output
    submission_output=$(qsub \
        -N "cadical_${base_name}" \
        -q "$PBS_QUEUE" \
        -l walltime=$PBS_WALLTIME \
        -v arg="$input_file $CADICAL_OPTIONS -t $CADICAL_TIMEOUT",out="$result_file",PBS_O_WORKDIR="$CADICAL_WORKDIR" \
        -o "$stdout_file" \
        -e "$stderr_file" \
        "$CADICAL_SCRIPT" 2>&1)

    # Extract job ID
    job_id=$(echo "$submission_output" | grep -oE '[0-9]+\.zeus-master')

    # Check if submission was successful
    if [[ -n "$job_id" ]]; then
        echo "Job ${base_name} submitted successfully with Job ID $job_id."
    else
        echo "Failed to submit job ${base_name}: $submission_output"
        echo "$input_file" >> "$failed_jobs_file"
    fi
}

# Function to get the next version of the failed jobs file
get_next_failed_jobs_file() {
    local run_dir=$1
    local version=1

    # Loop to find the next available version of failed_jobs file
    while [ -e "$run_dir/failed_jobs_v${version}.txt" ]; do
        version=$((version + 1))
    done

    echo "$run_dir/failed_jobs_v${version}.txt"
}

# Main execution
main() {
    local run_id
    local failed_jobs_file
    local run_dir

    # If the script is called with a failed jobs file
    if [ -n "$1" ]; then
        # Reuse the directory of the failed jobs file
        run_dir=$(dirname "$1")
        failed_jobs_file=$(get_next_failed_jobs_file "$run_dir")

        # Read failed jobs from the provided file
        input_files=()
        while IFS= read -r input_file; do
            input_files+=("$input_file")
        done < "$1"
    else
        # Create a new run directory
        run_id=$(get_timestamp)
        run_dir="$OUTPUT_DIR/$run_id"
        create_directory_structure "$run_dir"
        failed_jobs_file=$(get_next_failed_jobs_file "$run_dir")
        input_files=("$INPUT_DIR"/*.cnf)
    fi

    # Process each input file
    for input_file in "${input_files[@]}"; do
        base_name=$(get_base_filename "$input_file")
        read -r result_file stdout_file stderr_file < <(prepare_output_files "$run_dir" "$base_name")
        submit_job "$input_file" "$base_name" "$result_file" "$stdout_file" "$stderr_file" "$failed_jobs_file"
    done

    if [ -s "$failed_jobs_file" ]; then
        echo "Some jobs failed to submit. Check $failed_jobs_file to retry."
    else
        echo "All jobs submitted successfully."
    fi

    echo "Results will be in: $run_dir/"
}

# Run the main function, optionally using failed jobs as input
main "$1"
