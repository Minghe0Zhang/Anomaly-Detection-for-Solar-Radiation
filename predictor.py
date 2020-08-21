#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
"""
predictor for bernoulli point process
"""

class event_predictor():
    """
    A Space-Time series probability predictor given the testing sequence, this predictor could calculate the 
    possibility that
    """
    def __init__(self,beta_birth,beta_inter,testX):
        """
        beta_birth: learnt birthrate in each location, with shape [K]
        beta_inter: learnt interaction between pairs of locations, with shape [d,K,K]
        testX   : The sequence which is used for prediction, with shape [seq_len, k]
        """
        self.beta_birth = beta_birth
        self.beta_inter = beta_inter
        self.K          = beta_birth.shape[0]   # total number of locations
        self.d          = beta_inter.shape[0]   # memory depth
        self.testX      = testX                 #[seq_len, k]
        self.seq_len    = testX.shape[0]

    def prob_predict(self):
        """
        given a particular prediction result for the sequence in the form of possibility of an abnormal events
        output:
            prob_events: shape [seq_len - d, K]
        """
        # print(self.seq_len)
        # print(self.d)
        # print(self.beta_birth)
        # print(self.K)
        prob_events = np.zeros((self.seq_len-self.d, self.K))+ self.beta_birth # base intensity
        # evaluate the prob_events at time t given testX[t-d:t-1, k]
        for l0 in range(self.K):
            for s in range(self.d):
                for l1 in range(self.K):
                    prob_events[:,l0] += self.beta_inter[s,l0,l1]*self.testX[self.d-s:self.seq_len-s,l1]

        # prob_events = prob_events / np.max(prob_events)
        return prob_events

    def event_pred(self, prob_events,threshold=0.5):
        """
        given a particular event result for the sequence representing as 0 or 1
        input:
            prob_events: shape [seq_len - d, K] the output for function prob_predict()
        """ 
        return (prob_events > threshold)

    def accuracy_metric(self,threshold=0.5):
        """
        calculate precision, recall and F1 score. 
        """
        prob_events = self.prob_predict()
        pred_events = self.event_pred(prob_events, threshold=threshold) # [seq_len-d, K]
        TP          = np.sum(pred_events * self.testX[self.d:,:])  
        FP          = np.sum(pred_events * np.ones_like(self.testX[self.d:,:])) - TP
        # TN          = (self.seq_len-self.d)*self.K - TP
        FN          = np.sum(self.testX[self.d:,:]) - TP
        # print(TP)
        # print(FP)
        # print(self.testX.shape)
        precision   = TP / (TP+FP)
        recall      = TP / (TP+FN)
        F1          = 2 * (precision*recall)/(precision + recall)

        return precision, recall, F1
        

    def dynamic_th(self, prob_events,d2 = 10):
        """
        Setting the threshold dynamically based on the history
        prob_events: predicted intenisty [seq_len-d, K]
        self.textX : real observations   [seq_len,   K]
        d2: sliding window to 
        """
        d_th  = np.zeros((self.seq_len-self.d, self.K)) #[seq_len-d, K]
        for t in range(self.seq_len-self.d - d2):
            for k in range(self.K):
                count_abno = np.sum(self.testX[t+self.d:t+self.d+d2,k])
                d_th[t+d2,k] = np.sum(prob_events[t:t+d2,k]*self.testX[t+self.d:t+self.d+d2,k])/(count_abno)\
                             + np.sum(prob_events[t:t+d2,k]*(1-self.testX[t+self.d:t+self.d+d2,k]))/(d2-count_abno)
                d_th[t+d2,k] = d_th[t+d2,k] / 2

                # d_th[t+d2] = np.mean(prob_events[t:t+d2,k])

        return d_th

    def dynmamic_accuracy_metric(self):
        """
        calculate precision, recall and F1 score based on dynamic threshold 
        """
        prob_events = self.prob_predict()                   # [seq_len-d, K]
        d2          = 50
        d_th        = self.dynamic_th(prob_events,d2 = d2)  # [seq_len-d, K]
        pred_events = (prob_events > d_th)
        TP          = np.sum(pred_events[d2:] * self.testX[self.d+d2:,:])  
        FP          = np.sum(pred_events[d2:] * np.ones_like(self.testX[self.d+d2:,:])) - TP
        # TN          = (self.seq_len-self.d)*self.K - TP
        FN          = np.sum(self.testX[self.d+d2:,:]) - TP
        # print(TP)
        # print(FP)
        # print(self.testX.shape)
        precision   = TP / (TP+FP)
        recall      = TP / (TP+FN)
        F1          = 2 * (precision*recall)/(precision + recall)

        return precision, recall, F1


if __name__ == "__main__":
    print("test")