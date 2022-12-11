import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN


"""
INPUT: label data
OUTPUT: dict of MPAA rating counts
"""
def get_catagory_counts(y):
    return {
        "G": sum([1 if label == 1 else 0 for label in y]),
        "PG": sum([1 if label == 2 else 0 for label in y]),
        "Mature": sum([1 if label == 3 else 0 for label in y])
    }

# Random Under Sampling
def RUS(X, y):
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
    sm = SMOTE(sampling_strategy='auto')
    return sm.fit_resample(X, y.astype('int'))

# SMOTE + TomekLinks
def SMOTE_TL(X, y):
    smt = SMOTETomek(sampling_strategy='auto')
    return smt.fit_resample(X, y.astype('int'))

"""
For a set of data and a sampling type ('Imbalanced', 'RUS', 'TomekLinks', 'ENN', 'ROS', 'SMOTE', 'SMOTETomek')
performs the proper resampling
"""
def resample(X, y, sample_type="Imbalanced"):
    if sample_type == 'Imbalanced':
        return X,y
    if sample_type == 'RUS':
        return RUS(X, y)
    if sample_type == 'TomekLinks':
        return TLinks(X,y)
    if sample_type == 'ENN':
        return ENN(X,y)
    if sample_type == 'ROS':
        return ROS(X,y)
    if sample_type == 'SMOTE':
        return SMOTE_Reg(X,y)
    if sample_type == 'SMOTETomek':
        return SMOTE_TL(X,y)
    else:
        print(f'{sample_type} is not recognized')
        return None
