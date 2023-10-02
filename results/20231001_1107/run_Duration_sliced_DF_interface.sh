#!/bin/bash -x
for file in /app/results/20231001_1107/ASVspoof2021_DF_eval_part00_durations_*.txt;
do
    testName=$(basename $file .txt)
    echo $testName

    CUDA_VISIBLE_DEVICES=0 python Inference_SSL_DF.py --list_filename $file \
                           --model_path /app/model/Best_LA_model_for_DF.pth \
                           --eval_output=/app/results/20231001_1107/${testName}.res --audio_path /app/ASVspoof2021_DF_eval/flac/ 

    #bash ./meta2simpleEval.sh /app/SSL_Anti-spoofing/DF-keys-stage-1/keys/CM/DF_eval_part00_trial_metadata.txt /app/results/20231001_1107/${testName}.res 
    sed -i 's;.flac;;' ./evaluate_DF.py /app/results/20231001_1107/${testName}.res
    python ./evaluate_DF.py /app/results/20231001_1107/${testName}.res  /app/SSL_Anti-spoofing/DF-keys-stage-1/keys/CM/DF_eval_part00_trial_metadata.txt 
    cp roc.pdf /app/results/20231001_1107/${testName}_roc.pdf

done 
