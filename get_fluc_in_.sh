#!/bin/bash

#head Complete_trial_metadata.txt
#wait

# Get a list of all the files in the FLUC directory
fluc_files=$(ls /mnt/c/work/Github/Tensor_Lihu/MultiKol/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac)

# Loop through the meta file and extract the lines for the files that exist in the FLUC directory
while read line; do
  # Split the line into columns
  file_id=$(echo $line | awk '{print $2}')
  #echo $file_id

  # Check if the file exists in the FLUC directory
  if [[ -f "/mnt/c/work/Github/Tensor_Lihu/MultiKol/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/${file_id}.flac" ]]; then
    # Extract the line and print it to stdout
    echo "$line"
  fi
done < Complete_trial_metadata.txt
