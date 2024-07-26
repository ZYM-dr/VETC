#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022

@author: tianyu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def decision_curve_analysis(probs, y_test, p_min, p_max, epsilon):
    """
    Calculate the Net Benefit for a binary classifier
    :param probs: predicted probabilities
    :param y_test: Target vector (Test set)
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :return: Values for the threshold probabilities and their corresponding net benefit
    """

    p_serie = []
    net_benefit_serie = []
    for p in np.arange(p_min, p_max, epsilon):
        y_pred = probs[:, 1] > p
        y_pred = y_pred.astype('int16')
        y_pred[np.where(y_pred == 0)] = -1
        #print(y_pred)
        #print(confusion_matrix(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        n = tn + fp + fn + tp
        net_benefit = (tp / n) - (fp / n) * (p / (1 - p))
        p_serie.append(p * 100)
        net_benefit_serie.append(net_benefit)

    return p_serie, net_benefit_serie


# Net Benefit Prioritise All Referrals
def calculate_net_benefit_all(tp_test, tn_test, p_min, p_max, epsilon):
    """
    Calculate the Net Benefit for the 'Treat-All' clinical alternative
    :param tp_test: Number of True Positives in the Test set
    :param tn_test: Number of True Negative in the Test set
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :return: Values for the threshold probabilities and their corresponding net benefit
    """

    p_serie = []
    net_benefit_serie = []
    total = tp_test + tn_test
    for p in np.arange(p_min, p_max, epsilon):
        net_benefit = (tp_test / total) - (tn_test / total) * (p / (1 - p))
        p_serie.append(p * 100)
        net_benefit_serie.append(net_benefit)

    return p_serie, net_benefit_serie


def plot_decision_curves(prob_list, labels,  y_test, p_min, p_max, epsilon, net_benefit_lower, net_benefit_upper):
    """
    Plotting the Net Benefit for a List of Classifiers (scikit-learn)
    :param clfs: List of binary classifiers
    :param labels: List of names (for including in the graph legend)
    :param X_test: Independent features (Test set)
    :param y_test: Target vector (Test set)
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :param net_benefit_lower: Lower limit for the Net Benefit (y axis)
    :param net_benefit_upper: Upper limit for the Net Benefit (y axis)
    :return: Decision Curve Analysis Plot
    """
    # Calculating True Positives and True Negatives (Test Set)
    # tp_test = np.sum(y_test)
    # tn_test = y_test.shape[0] - tp_test
    tp_test = np.sum(y_test == 1)
    tn_test = np.sum(y_test == -1)

    # Defining Figure Size
    plt.figure(figsize=(15, 10), dpi=80)

    # Plotting each Classifier
    for i in range(0, len(prob_list)):
        p, net_benefit = decision_curve_analysis(prob_list[i], y_test, p_min, p_max, epsilon)
        plt.plot(p, net_benefit, label=labels[i])

    # Plotting Prioritise None
    plt.hlines(y=0, xmin=p_min * 100, xmax=p_max * 100, label='Treat None', linestyles='dotted', color='red')

    # Plotting Prioritise All
    p_all, net_benefit_all = calculate_net_benefit_all(tp_test, tn_test, p_min, p_max, epsilon)
    plt.plot(p_all, net_benefit_all, label='Treat All', linestyle='dashed', color='black')

    # Figure Configuration
    plt.xlabel('Threshold Probability in %', )
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.ylim([net_benefit_lower, net_benefit_upper])
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
    plt.legend()

    return plt.show()