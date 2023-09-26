#!/bin/bash

inMetaFile=$1
resFile=$2

# Create a temporary file to store the matched lines from meta.csv
temp_file=$(mktemp)

# Iterate over the lines in results.res
while read -r line; do

  # Extract the file name from the line
  file_name=$(cut -d' ' -f1 <<< "$line")

  # Search for the matching line in meta.csv
  matching_line=$(grep -e ^$file_name $inMetaFile)

  # If a matching line is found, append it to the temporary file
  if [[ -n "$matching_line" ]]; then
    echo "$matching_line" >> "$temp_file"
  fi

done <  $resFile

# Print the 1st and 3rd columns of the matched lines, space separated
while read -r line; do
  echo "$line" |  cut -d',' -f1,3 | sed 's/,/ /' | sed 's/bona-fide/bonafide/'
done < "$temp_file"

# Remove the temporary file
rm "$temp_file"
