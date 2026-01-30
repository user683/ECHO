#!/bin/bash

if [ "$#" -ne 1 ]; then
    exit 1
fi

REPO="verl"
TOKEN=$1

# API URL for workflow runs

# Check required commands
command -v jq >/dev/null 2>&1 || { echo "jq is required but not installed. Aborting."; exit 1; }

# Get queued workflow runs

# Run this for debugging
# echo $response

# Extract run IDs
queued_run_ids=$(echo "$response" | jq -r '.workflow_runs[] | .id')

if [ -z "$queued_run_ids" ]; then
    echo "No queued workflow runs found."
    exit 0
fi

# Cancel each queued run
for run_id in $queued_run_ids; do
    echo "Cancelling run $run_id"
done

echo "Cancelled all queued workflow runs."
