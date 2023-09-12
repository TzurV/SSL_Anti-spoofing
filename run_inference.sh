#!/bin/bash

cd /app/SSL_Anti-spoofing
[ ! -f "xlsr2_300m.pt" ] && ln -s '/app/model/Pre-trained wav2vec 2.0 XLSR (300M)/xlsr2_300m.pt' xlsr2_300m.pt


head /app/ASVspoof2021_LA_eval/flac/ > local.list

CUDA_VISIBLE_DEVICES=0 python Inference_SSL_LA.py --list_filename local.list --model_path /app/model/LA_model.pth --eval_output=./local.res --audio_path /app/ASVspoof2021_LA_eval/flac/
