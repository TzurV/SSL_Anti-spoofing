#!/usr/bin/env python
"""
Script to compute pooled EER for ASVspoof2021 DF. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol.
    Please follow README, download the key files, and use ./keys
Example:
$: python evaluate.py score.txt ./keys
"""
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob

if len(sys.argv) != 3:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
key_file = sys.argv[2]
#phase = sys.argv[3]

#cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')
#print(f"CM key file: {key_file}")


def eval_to_score_file(score_file, cm_key_file):
    
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    print(submission_scores.shape)
    #if len(submission_scores) != len(cm_data):
    #    print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
    #    exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
        exit(1)
            
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    print(cm_data.shape)
    if len(cm_data.columns)>2:
        # assume metadata.txt
        '''
        example of cm_scores after merge
                    0       1_x           1_y         5
        0  LA_E_9998106 -6.127488  LA_E_9998106     spoof
        1  LA_E_6458354 -6.044828  LA_E_6458354     spoof
        2  LA_E_1195666 -6.125190  LA_E_1195666     spoof
        3  LA_E_4485249 -6.158697  LA_E_4485249  bonafide
        '''

        cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=1, how='inner')  # check here for progress vs eval set
        print(cm_scores.shape)
        print(cm_scores.head())
    
        bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
        spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values

    else:
        '''
        simplified input with 2 columns
        LA_E_9332881 spoof
        LA_E_6866159 spoof
        LA_E_5464494 spoof
        LA_E_4759417 spoof        

        example of cm_scores after merge
                    0       1_x       1_y
        0  LA_E_9998106 -6.127488     spoof
        1  LA_E_6458354 -6.044828     spoof
        2  LA_E_1195666 -6.125190     spoof
        3  LA_E_4485249 -6.158697  bonafide
        4  LA_E_9773731 -6.060637     spoof
        '''
        cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=0, how='inner')  # check here for progress vs eval set
        print(cm_scores.shape)
        print(cm_scores.head())

        bona_cm = cm_scores[cm_scores['1_y'] == 'bonafide']['1_x'].values
        spoof_cm = cm_scores[cm_scores['1_y'] == 'spoof']['1_x'].values


    #cm_scores = submission_scores.merge(cm_data[cm_data[7] == 'progress'], left_on=0, right_on=1, how='inner')  # check here for progress vs eval set
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100*eer_cm)
    print(out_data)
    return eer_cm

if __name__ == "__main__":

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)
        
    if not os.path.isfile(key_file):
        print("%s doesn't exist" % (key_file))
        exit(1)

    #if phase != 'progress' and phase != 'eval' and phase != 'hidden_track':
    #    print("phase must be either progress, eval, or hidden_track")
    #    exit(1)

    _ = eval_to_score_file(submit_file, key_file)
