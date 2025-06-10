#!/bin/bash


SEARCH_STRING=${2:-"checkpoint"}
PARENT_STRING=${3:-"0"}
TARGET_DIRECTORY=${1:-"saves/"}

find "$TARGET_DIRECTORY" -name "*$SEARCH_STRING*" | while read -r file; do
    parent_dir=$(dirname "$file")
    
    if [[ $PARENT_STRING == "0" ]]; then
        echo "Removing: $file"
        rm -rf "$file" 
    fi
    if [[ $PARENT_STRING != "0" ]]; then
        if [[ $parent_dir != *"$PARENT_STRING"* ]]; then
            echo "Removing: $file"
            rm -rf "$file"
        fi
    fi
done

# bash config/rm.sh saves0/ checkpoint
# bash config/rm.sh saves0/ safetensors 4-