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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


if len(sys.argv) != 3:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
key_file = sys.argv[2]
#phase = sys.argv[3]

#cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')
#print(f"CM key file: {key_file}")


def plot_circle(x, y, value, color='red'):
  """Plots a circle at the closest point to a given value in a sorted x,y plot.

  Args:
    x: A NumPy array containing the x-values of the plot.
    y: A NumPy array containing the y-values of the plot.
    value: The value to plot the circle at.
    color: The color of the circle.
  """

  # Find the closest index to the given value.
  closest_index = np.argmin(np.abs(x - value))

  # Plot the circle at the closest point.
  plt.plot(x[closest_index], y[closest_index], 'o', color=color)

def find_eer(fpr, fnr):
  """Finds the equal error rate (EER).

  Args:
    fpr: A NumPy array containing the false positive rates.
    fnr: A NumPy array containing the false negative rates.

  Returns:
    A float containing the equal error rate.
  """

  eer = (fpr + fnr) / 2
  return min(eer)

def draw_roc(fpr, tpr, auc_value, eer, thresh, save_path):
  """Draws the ROC curve into a PDF.

  Args:
    fpr: A NumPy array containing the false positive rates.
    tpr: A NumPy array containing the true positive rates.
    auc: A float containing the area under the ROC curve.
    eer: A float containing the equal error rate.
    save_path: A string containing the path to save the PDF to.
  """

  fig, ax = plt.subplots()
  ax.plot(fpr, tpr, label='ROC curve (AUC = {:.3f}, eer = {:.3f})'.format(auc_value, eer))
  ax.plot([0, 1], [0, 1], 'k--', label='Random')
  #ax.plot([eer], [eer], 'o', color='red', label='EER ({:.3f})'.format(eer))
  plot_circle(fpr, tpr, eer)
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('ROC Curve')
  ax.legend()
  plt.savefig(save_path)


def eval_to_score_file(score_file, cm_key_file):
    
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    print(f"Submission scores shape: {submission_scores.shape}")
    #print(submission_scores)
    #if len(submission_scores) != len(cm_data):
    #    print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
    #    exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
        exit(1)
            
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    print(f"CM data shape: {cm_data.shape}")
    print(cm_data)
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

        # Load the DataFrame
        df = cm_scores #pd.read_csv('data.csv')

        # Get the class labels and scores
        y = df[5]
        x = df['1_x']

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y, x, pos_label='bonafide')

        # Calculate the area under the ROC curve (AUC)
        area_under_ROC = auc(fpr, tpr)

        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        # Find the equal error rate (EER)
        #eer = find_eer(fpr, tpr)

        # Draw the ROC curve into a PDF
        print("TV:DBG save ROC curve")
        draw_roc(fpr, tpr, area_under_ROC, eer, thresh, 'roc.pdf')

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

        # Load the DataFrame
        df = cm_scores #pd.read_csv('data.csv')

        # Get the class labels and scores
        y = df['1_y']
        x = df['1_x']

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y, x, pos_label='bonafide')

        # Calculate the area under the ROC curve (AUC)
        area_under_ROC = auc(fpr, tpr)

        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        # Find the equal error rate (EER)
        #eer = find_eer(fpr, tpr)

        # Draw the ROC curve into a PDF
        print("TV:DBG save ROC curve")
        draw_roc(fpr, tpr, area_under_ROC, eer, thresh, 'roc.pdf')

    #cm_scores = submission_scores.merge(cm_data[cm_data[7] == 'progress'], left_on=0, right_on=1, how='inner')  # check here for progress vs eval set
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.4f\n" % eer_cm
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
