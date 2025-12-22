#/usr/bin/env bash

set -eo pipefail

exit_venv() {
    if [ "$did_set_venv" -eq "1" ]; then
        deactivate && echo "venv deactivated" || echo "failed to deactivate venv (sorry!)"
    fi
}

sigint_handler() {
    echo "Caught SIGINT, cleaning up..."
    kill -9 $(jobs -p)
    echo "Done"
    exit_venv
}

trap sigint_handler SIGINT
trap sigint_handler SIGTERM

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            FILENAME="$2"
            shift
            shift
            ;;
        -w|--world-size)
            WORLD_SIZE="$2"
            shift
            shift
            ;;
    esac
done

if [ -z $FILENAME -o -z $WORLD_SIZE ]; then
    echo "No world size or no filename provided"
    exit 1
fi

did_set_venv="0"
if [ -z $VIRTUAL_ENV ]; then
    read -p "You haven't sourced a virtual environment, would you like to do so? (y/n): " choice
    choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
    if [[ -z "$choice" ]]; then
        choice="y"
    fi

    case "$choice" in
        y|yes)
            read -p "Please input path to the venv folder (if omitted, uses ./.venv): " venv_path
            ;;
    esac

    if [[ -z "$venv_path" ]]; then
        venv_path="./.venv"
    fi

    if [ -z "$venv_path" ]; then
        echo "No such path: $venv_path"
        exit 1
    fi

    source "$venv_path/bin/activate"
    did_set_venv="1"
fi

LOGS_PATH="./logs"
if [ ! -d $LOGS_PATH ] ; then
    echo "Creating logs directory $LOGS_PATH"
    mkdir -p "$LOGS_PATH"
fi

for (( RANK=0; RANK < WORLD_SIZE; RANK++ )); do
    python3 $FILENAME --rank="$RANK" --world_size="$WORLD_SIZE" &
done

echo "Waiting for all processes to exit. Hit Ctrl-C if you want to stop execution."
wait

exit_venv
