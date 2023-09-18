#!/bin/bash

cd /app/SSL_Anti-spoofing
[ ! -f "xlsr2_300m.pt" ] && ln -s '/app/model/Pre-trained wav2vec 2.0 XLSR (300M)/xlsr2_300m.pt' xlsr2_300m.pt

mkdir -p shorteval/ASVspoof_LA_cm_protocols
evalFile=shorteval/ASVspoof_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt
grep progress /app/ASVspoof2021_LA_eval/keys/LA/ASV/Original_trial_metadata.txt |\
     grep 'LA2021' | awk 'BEGIN {srand()} (rand() <= 0.01){A[$6]+=1; print $2;}' | sed 's/LA2021-//' > $evalFile
 
wc -l shorteval/ASVspoof_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt

for ID in `awk '{ sub("\r$", ""); print }' $evalFile`; do 
    grep $ID /app/ASVspoof2021_LA_eval/keys/LA/ASV/Original_trial_metadata.txt |\
    grep progress | head -n 1;
done > /app/ASVspoof2021_LA_eval/keys/LA/ASV/trial_metadata.txt

formatted_date=$(date +"%Y%m%d_%H%M")
echo $formatted_date

resultsPath=/app/results/$formatted_date
mkdir -p $resultsPath
cp /app/ASVspoof2021_LA_eval/keys/LA/ASV/trial_metadata.txt $resultsPath
cp $evalFile $resultsPath

for ID in `awk '{ sub("\r$", ""); print }' $evalFile`; do grep $ID /app/ASVspoof2021_LA_eval/keys/LA/ASV/ASVTorch_Kaldi/Original_score.txt | head -n 1; done > /app/ASVspoof2021_LA_eval/keys/LA/ASV/ASVTorch_Kaldi/score.txt 

CUDA_VISIBLE_DEVICES=0 python main_SSL_LA.py --track=LA --is_eval --eval --model_path=/app/model/LA_model.pth --eval_output=$resultsPath/eval_CM_scores_file_SSL_LA.txt --database_path /app/ --protocols_path shorteval/
python evaluate_2021_LA.py /app/results/eval_CM_scores_file_SSL_LA.txt /app/ASVspoof2021_LA_eval/keys/LA/ progress

#CUDA_VISIBLE_DEVICES=0 python main_SSL_DF.py --track=DF --is_eval --eval --model_path=/app/model/Best_LA_model_for_DF.pth --eval_output=$resultsPath/eval_CM_scores_file_SSL_Best_LA_model_for_DF.txt --database_path /app/ --protocols_path shorteval/
#python evaluate_2021_DF.py /app/results/eval_CM_scores_file_SSL_Best_LA_model_for_DF.txt /app/ASVspoof2021_LA_eval/keys/LA/ progress
