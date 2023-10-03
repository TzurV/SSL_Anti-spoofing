#!/bin/bash

cd /app/SSL_Anti-spoofing
[ ! -f "xlsr2_300m.pt" ] && ln -s '/app/model/Pre-trained wav2vec 2.0 XLSR (300M)/xlsr2_300m.pt' xlsr2_300m.pt

formatted_date=$(date +"%Y%m%d_%H%M")
echo $formatted_date

resultsPath=/app/results/$formatted_date
mkdir -p $resultsPath


mkdir -p shorteval/ASVspoof_DF_cm_protocols
evalFile=shorteval/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt

sourceTrialMetadata=/app/SSL_Anti-spoofing/DF-keys-stage-1/keys/CM/DF_eval_part00_trial_metadata.txt
trialMetadataDir=/app/ASVspoof2021_DF_eval/keys
mkdir -p $trialMetadataDir/CM
trialMetadata=/app/ASVspoof2021_DF_eval/keys/CM/trial_metadata.txt


grep progress $sourceTrialMetadata |\
     awk 'BEGIN {srand()} (rand() <= 0.05){A[$6]+=1; print $2;}'  > $evalFile
 
wc -l $evalFile

for ID in `awk '{ sub("\r$", ""); print }' $evalFile`; do 
    grep $ID $sourceTrialMetadata |\
    grep progress | head -n 1;
done > $trialMetadata

cp $trialMetadata $resultsPath
cp $evalFile $resultsPath

time CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py --track=DF --is_eval --eval --model_path=/app/model/Best_LA_model_for_DF.pth --eval_output=$resultsPath/eval_CM_scores_file_SSL_Best_LA_model_for_DF.txt --database_path /app/ --protocols_path shorteval/
python evaluate_2021_DF.py $resultsPath/eval_CM_scores_file_SSL_Best_LA_model_for_DF.txt $trialMetadataDir progress

echo "python evaluate_DF.py $resultsPath/eval_CM_scores_file_SSL_Best_LA_model_for_DF.txt $trialMetadata"
python evaluate_DF.py $resultsPath/eval_CM_scores_file_SSL_Best_LA_model_for_DF.txt $trialMetadata
