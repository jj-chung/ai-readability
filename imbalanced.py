import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

# Random Under Sampling
def RUS(X, y):
    # sampling_dict =
    rus = RandomUnderSampler(sampling_strategy='auto')
    return rus.fit_resample(X, y.astype('int'))

# TomekLinks Under Sampling
def TLinks(X,y):
    tl = TomekLinks(sampling_strategy='auto')
    return tl.fit_resample(X, y)

# ENN Under Sampling
def ENN(X,y):
    en = EditedNearestNeighbours(sampling_strategy='auto')
    return en.fit_resample(X, y)

# Random Over Sampling
def ROS(X, y):
    ros = RandomOverSampler(sampling_strategy='auto')
    return ros.fit_resample(X, y.astype('int'))

# SMOTE
def SMOTE_Reg(X, y):
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    return sm.fit_resample(X, y.astype('int'))

# SMOTE + TomekLinks
def SMOTE_TL(X, y):
    smt = SMOTETomek(sampling_strategy='auto', random_state=42)
    return smt.fit_resample(X, y.astype('int'))
