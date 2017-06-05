import numpy as np
import pandas as pd
from sklearn import model_selection as skSel

def mirror(log, imgcol, numcol, appendStr='_mirror'):
    ''' Returns a copy of log with 2x the records by appending records where
     the imgcol values are appended with appendStr and numcol values are negated.
    '''
    mirror = log.copy()
    mirror[imgcol] = mirror[imgcol] + appendStr
    mirror[numcol] = -mirror[numcol].astype(np.float32)
    return pd.concat([log, mirror], axis=0, ignore_index=True)

def filter_gte(log, col, val, verbose=True):
    ''' Removes record from log if col's value is < val
    '''
    cnt = log.shape[0]
    log = log[log[col] >= val]
    if verbose:
        print("%d records have been removed due to speed <= %d" % (cnt - log.shape[0], val))
    return log

def shuffle(log):
    return log.reindex(np.random.permutation(log.index))

def train_test_split(log, test_size=.2):
    itrain, itest = skSel.train_test_split(np.arange(log.shape[0]), test_size=test_size)
    train_set = log.iloc[itrain]
    test_set = log.iloc[itest]
    return (train_set, test_set)