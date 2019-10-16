import numpy as np
import pandas as pd

import gc
from datetime import datetime
from numba import jit
import os

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

gl_reg_folder = '../../model'
gl_reg_file_name = 'model_register.csv'

gl_model_type = 'lgbm'
gl_model_obj = 'regr' # class  - for classification , regr  - for regression
gl_model_eval_func = 'mae'
gl_model_name = 'bs'
gl_featset_name = 'bs'
gl_seed = 42
np.random.seed(gl_seed)
gl_fold = 5

gl_target_name = 'FinalPlannedRevenueSynergy'#'AnnouncedCostSynergy'
gl_id_name = 'Id'

gl_params = {'model_type': gl_model_type # 'lgbm'
            ,'model_obj': gl_model_obj # 'regr' - regression
            ,'model_name': gl_model_name
            ,'featset_name': gl_featset_name
            ,'model_eval_func': gl_model_eval_func
            ,'target_name': gl_target_name
            ,'id_name': gl_id_name
            ,'fold': gl_fold
            ,'seed': gl_seed
            ,'reg_folder': gl_reg_folder
            ,'reg_file_name': gl_reg_file_name
            }

#Fix for wrong parcing during L2 csv import 
def parse_fix(df):    
    """Fixing wrong parcing of Initiative description. """
    #Convert Value to numeric, strings will became NaN
    df['Value'].fillna(0, inplace=True)
    df['Value']=df['Value'].apply(pd.to_numeric, errors='coerce')
    #Shift left rows where description overlapped Value
    while df['Value'].isna().sum()>0:
        rows=df['Value'].isna()
        cols=slice('Value','SubFunction')
        df.loc[rows, cols]=df.loc[rows, cols].apply(lambda x: x.shift(-1,fill_value=''), axis=1)
    #replace both - real newlines,tabs, etc. and literal
    df.replace(to_replace=[r'\\t|\\n|\\r', '\t|\n|\r'], value=['',''], regex=True, inplace=True)
    #return df

#Remove newlines, tabs, carriage returns in csv after pandas parse   
def fixnewlines(csvfile):
    """Replace newlines, tabs, carriage returns after csv parsed by pandas and overwrite by corrected csv"""
    df = pd.read_csv(csvfile)
    #replace both - real newlines,tabs, etc. and literal
    df.replace(to_replace=[r'\\t|\\n|\\r', '\t|\n|\r'], value=['',''], regex=True, inplace=True)
    df.to_csv(csvfile, index=False)    
    
#Сommon functions for exploratory data analysis
def get_stats(df):
    """
    Function returns a dataframe with the following stats for each column of df dataframe:
    - Unique_values
    - Percentage of missing values
    - Percentage of zero values
    - Percentage of values in the biggest category
    - data type
    """
    stats = []
    for col in df.columns:
        if df[col].dtype not in ['object', 'str', 'datetime64[ns]']:
            zero_cnt = df[df[col] == 0][col].count() * 100 / df.shape[0]
        else:
            zero_cnt = 0

        stats.append((col, df[col].nunique(),
                      df[col].isnull().sum() * 100 / df.shape[0],
                      zero_cnt,
                      df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                      df[col].dtype))

    df_stats = pd.DataFrame(stats, columns=['Feature', 'Unique_values',
                                            'Percentage of missing values',
                                            'Percentage of zero values',
                                            'Percentage of values in the biggest category',
                                            'type'])
    # df_stats.sort_values('Percentage of zero values', ascending=False, inplace=True)

    del stats
    gc.collect()

    return df_stats


def drop_zero_cols(df, stats, prc):
    """
    Function drops all columns in df dataframe where percentage of zero values >= prc value.
    Percentage of zero values per column is taken from stats dataframe.
    """
    cols_to_drop = list(stats[stats['Percentage of zero values'] >= prc]['Feature'])
    for i in cols_to_drop:
        df.drop([i], axis=1, inplace=True)
    print(f'{len(cols_to_drop)} columns with zero values % >= {prc} have been dropped')


def drop_missing_cols(df, stats, prc):
    """
    Function drops all columns in df dataframe where percentage of missing values >= prc value.
    Percentage of missing values per column is taken from stats dataframe.
    """
    cols_to_drop = list(stats[stats['Percentage of missing values'] >= prc]['Feature'])
    for i in cols_to_drop:
        df.drop([i], axis=1, inplace=True)
    print(f'{len(cols_to_drop)} columns with missing values % >= {prc} have been dropped')


def corr_heatmap(df):
    """
    Function generates a correlation heatmap plot for features of df dataframe.
    """
    corr_matrix=df.corr().abs()  
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(12,10))  
    sns.heatmap(corr_matrix, 
                mask= mask,
                cmap='coolwarm', 
                square=True,
                linewidths=0.5,            
                annot=True, fmt=".2f",
                cbar_kws={'label': 'Correlation',"shrink": 1},
                ax=ax
               )

    
    

# Encoding functions

# If you call encode_TE, encode_FE on training data then the function
# returns a 2 element python list containing [list, dictionary]
# the return[0] = list are the names of new columns added
# the return[1] = dictionary are which category variables got encoded
#
# In case of encode_NE the function
# returns a 2 element python list containing [list, dictionary]
# the return[0] = list are the names of new columns added
# the return[1] = dictionary are with which mean and std variables got encoded
#
# When encoding prediction data after one of 3 calls above, use 'encode_?E_prd'
# and pass the dictionary. If you don't use one of 3 above, then you can
# call basic 'encode_?E' on prediction data.

# CHECK FOR NAN
def nan_check(x):
    if isinstance(x, float):
        if np.isnan(x):
            return True
    return False

# TARGET ENCODING
def encode_TE(df, col, tar=gl_target_name):
    d = {}
    v = df[col].unique()
    for x in v:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col] == x].mean()
        d[x] = m
    n = col + "_TE"
    df[n] = df[col].map(d)
    return [[n], d]

# TARGET ENCODING from dictionary
# xx can be e.g.:
# - for classification : 0.5
# - for regression : np.mean(list(mp.values())) (mean of all calculated means in train)
def encode_TE_prd(df, col, mp, xx=0):
    n = col + "_TE"
    df[n] = df[col].map(mp).fillna(xx)
    return [n]

# TARGET ENCODING for K-fold cross validation training
def encode_TE_fold(df, col, fold=gl_fold, seed=gl_seed, tar=gl_target_name):
    """
    Function generates target encoded col_TE column for input col column in df dataframe, using K-fold process.
    In each fold it does the following:
    - splits df data into trn (data from K-1 folds) and val (out of hold data from the remaining fold);
    - calculates target mean for each unique value of column col in trn;
    - maps calculated target means to correspoding values in val.
    This approach helps to avoid overfitting and data leak, leading to overfitting.
    Note: the approach implies that target encoding and then training process use the same settings for K-Fold,
    namely: number of folds and seed.
    """
    n = col + "_TE"
    df[n] = np.nan
    y = df[tar]
    d = []

    folds = KFold(n_splits=fold, shuffle=True, random_state=seed)
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn = df.iloc[trn_]
        val = df.iloc[val_]
        means = trn.groupby(col)[tar].mean()
        df[n].iloc[val_] = val[col].map(means).fillna(means.mean())

    del trn, val, y
    gc.collect()

    return [n]

# FREQUENCY ENCODING
def encode_FE(df, col):
    d = df[col].value_counts(dropna=False)
    n = col + "_FE"
    df[n] = df[col].map(d) / d.max()
    return [[n], d]

# FREQUENCY ENCODING from dictionary
def encode_FE_prd(df, col, mp, xx=1.0):
    cv = df[col].value_counts(dropna=False)
    n = col + "_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(mp), n] = xx * np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    return [[n], mp]

# NUMERIC ENCODING
def encode_NE(df, col):
    n = col + "_NE"
    df[n] = df[col].astype(float)
    mx = np.std(df[n])
    mn = df[n].mean()
    df[n] = (df[n].fillna(mn) - mn) / mx
    return [[n], [mn, mx]]

# NUMERIC ENCODING from mean and std
def encode_NE_prd(df, col, mm):
    n = col + "_NE"
    df[n] = df[col].astype(float)
    df[n] = (df[n].fillna(df[n].mean()) - mm[0]) / mm[1]
    return [[n], mm]

# LABEL ENCODING
def encode_LE(df, col):
    le = LabelEncoder().fit(df[col])
    n = col + "_LE"
    df[n] = le.transform(df[col])
    return [n]

#TRANSFORMER FUNCTIONS

def transform_log1p(df, cols):
    '''Function will perform log1p transformation for inputed columns'''
    #list of columns for origingal features np.log1p == log(1 + x) transformation
    cols_log1p=[]
    
    # np.log1p transformation
    for col in cols:
        col_lg=col+'_Log1p'
        cols_log1p.append(col_lg)
        df[col_lg]=np.log1p(df[col])  

    return cols_log1p

def transform_log10(df, cols):
    '''Function will perform log10 transformation for inputted columns'''
    #list of columns for origingal features np.log10 transformation
    cols_log10=[]
    
    # np.log10 transformation
    for col in cols:
        col_lg=col+'_Log10'
        cols_log10.append(col_lg)
        df[col_lg]=np.log10(df[col])  

    return cols_log10

'''
# Various examples of applying encoding
cols = []

# NUMERIC ENCODE
cols += encode_NE(l1,'AcquirerRevenue')[0]

#FREQUENCY ENCODE
cols += encode_FE(l1,'Year')[0]

#TARGET ENCODE
cols += encode_TE(l1,'Year')[0]

print(f"New created columns are {cols}")

f = ['AcquirerRevenue','AcquirerRevenue_NE','TargetRevenue','Year','Year_FE', 'Year_TE']
l1[f].head()
'''

'''
Example of numeric encoding of train data when we save encoding dictionary;
we can use  this dictionary to encode prediction data later.

l1_prd = l1.copy()
tmp = encode_NE(l1,'TargetRevenue')
cols += tmp[0]; dictTargetRevenue = tmp[1]
cols_prd = []
cols_prd += encode_NE_prd(l1_prd,'TargetRevenue', dictTargetRevenue)[0]
f = ['AcquirerRevenue','AcquirerRevenue_NE','TargetRevenue','TargetRevenue_NE','Year','Year_FE', 'Year_TE']
l1_prd[f].head()
'''

'''
# LABEL ENCODING
for col in ['GeographicOrigin', 'Industry', 'SubIndustry']:
    cols += encode_LE(l1,col)
'''

'''
Example of numeric encoding of train data when we save encoding dictionary;
we can use  this dictionary to encode prediction data later.

l1_prd = l1.copy()
tmp = encode_NE(l1,'TargetRevenue')
cols += tmp[0]; dictTargetRevenue = tmp[1]
cols_prd = []
cols_prd += encode_NE_prd(l1_prd,'TargetRevenue', dictTargetRevenue)[0]
f = ['AcquirerRevenue','AcquirerRevenue_NE','TargetRevenue','TargetRevenue_NE','Year','Year_FE', 'Year_TE']
l1_prd[f].head()
'''

#Functions for training and prediction output management
def gen_file_name(dt, score, file_name_prefix):
    return file_name_prefix + '_' + str(round(score, 6)).replace('.', '') + '_' + str(dt.strftime('%Y%m%d%H%M'))

def save_prediction(df_, folder, file_name, reg_folder=gl_reg_folder, prediction_type = 'trn'):
    """Function saves df_ dataframe with predictions to file file_name_prediction_type.csv in folder reg_folder/folder"""
    # prediction_type can be 'trn' - train (oof), 'prd' - prediction
    if not os.path.exists(reg_folder + '/' + folder):
        os.mkdir(reg_folder + '/' + folder)
    df_.to_csv(reg_folder + '/' + folder + '/' + file_name + '_' + prediction_type + '.csv', index=False, float_format='%.6f')
    print('\nSaved to {}, shape={}'.format(file_name + '_' + prediction_type, df_.shape))


def register_prediction(date, folder, file_name, featset_name, score, reg_folder=gl_reg_folder, \
                        reg_file_name=gl_reg_file_name):
    """Function registers prediction file file_name in global register file reg_file_name.csv in folder reg_folder"""
    df_reg = pd.read_csv(reg_folder + '/' + reg_file_name)
    df_reg.loc[df_reg.shape[0]] = [date, folder, file_name, featset_name, round(score, 6)]
    df_reg.to_csv(reg_folder + '/' + reg_file_name, index=False)


# Custom and optimized model evaluation functions
@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(dtrain, preds):
    return fast_auc(dtrain, preds)


#Class for model output management
class Model():
    """
    Model class provides a group of methods to save and load trained models, their parameters and dictionaries.
    """

    def __init__():
        pass

    def save_mdl(self):
        """
        Method saves a model to a pickle file.
        """
        with open(self.file_name + ".pkl", 'wb') as file:
            pickle.dump(self.mdls, file)

    def load_mdl(self):
        """
        Method loads a model from a pickle file.
        """
        self.mdls = None
        with open(self.file_name + ".pkl", 'rb') as file:
            self.mdls = pickle.load(file)


#Class for training and prediction
class Trainer(Model):
    """
    Class Trainer provides q group of methods to preprocess data, train models and generate predictions.
    """

    def __init__(self, trn_file, prd_file, fe_trn, fe_prd, model_type, model_obj, model_name, featset_name,
                 model_eval_func, model_params, target_name, id_name, fold, seed, reg_folder, reg_file_name,
                 save_importances='n'):
        """
        Method initializes an instance of class Trainer.
        """
        self.trn_file = trn_file
        self.prd_file = prd_file
        self.file_name_prefix = model_type + '_' + model_obj + '_' + model_name + '_' + featset_name
        self.model_type = model_type
        self.featset_name = featset_name
        self.col_target = target_name
        self.col_id = id_name
        self.fill_nan = None
        self.importances = None
        self.features = None
        self.score = None
        self.fold = fold
        self.seed = seed
        self.fe_trn = fe_trn
        self.fe_prd = fe_prd
        self.reg_folder = reg_folder
        self.reg_file_name = reg_file_name
        # to set up scoring parameters
        self.metrics_dict = {'mae': {'metric_name': 'mae',
                                     'sklearn_scoring_function': metrics.mean_absolute_error},
                             'mse': {'metric_name': 'mse',
                                     'sklearn_scoring_function': metrics.mean_squared_error},
                             'auc': {'metric_name': eval_auc,
                                     'sklearn_scoring_function': eval_auc}  # fix to speed up calculation
                             }

        if model_type == 'lgbm':
            if model_obj == 'class':
                self.mdl = lgb.LGBMClassifier(**model_params)
            else:
                self.mdl = lgb.LGBMRegressor(**model_params)
        self.mdls = None
        self.eval_func = model_eval_func
        self.save_importances = save_importances

    def _get_data_trn(self):
        """
        Method does the following steps for training data:
        - reads train data from trn_file;
        - preprocesses train data, using function fe_trn;
        - splits train data into:
            train_x - dataframe with all features to train a model on;
            train_y - dataframe with labels (targets)
            train_id - dataframe with identifiers for each row of train data set.
        """
        df = pd.read_csv(self.trn_file)

        # replace nans, generate new features, do encoding etc.
        # outputs are:
        # - selected features to train a model on
        # - fill_nan - dataframe with features and mean values to replace nans in prediction dat set
        # - dict_te - dictionary for encoded features with values and target encodings to apply it to predicition data set
        self.features, self.fill_nan, self.dict_te = self.fe_trn(df
                                                                 , target=self.col_target
                                                                 , fold=self.fold
                                                                 , seed=self.seed)
        self.train_x = df[self.features]
        self.train_y = df[self.col_target]
        self.train_id = df[self.col_id]

        del df
        gc.collect()

    # get and preprocess prediction data set
    def _get_data_prd(self):
        """
        Method does the following steps for prediction data:
        - reads train data from prd_file;
        - preprocesses prediction data, using function fe_prd;
        - splits train data into:
            prd_x - dataframe with all features to build prediction, using pretrained model;
            prd_id - dataframe with identifiers for each row of train data set.
        """
        df = pd.read_csv(self.prd_file)

        # replace nans, generate new features, do encoding etc. using encodings, generated during the training phase
        # outputs are:
        # - selected features to train a model on
        self.features = self.fe_prd(df
                                    , fill_nan=self.fill_nan
                                    , dict_te=self.dict_te
                                    , target=self.col_target
                                    , fold=self.fold
                                    , seed=self.seed)

        self.prd_x = df[self.features]
        self.prd_id = df[self.col_id]

        del df
        gc.collect()

    def _save_importances(self):
        """
        Method outputs feature importance plot, based on features importances, generated during training phase.
        If save_importances = 'y', it saves the plot to a file.
        """
        mean_gain = self.importances[['gain', 'feature']].groupby('feature').mean()
        self.importances['mean_gain'] = self.importances['feature'].map(mean_gain['gain'])
        plt.figure(figsize=(8, 6))
        sns.barplot(x='gain', y='feature', data=self.importances.sort_values('mean_gain', ascending=False))
        plt.tight_layout()
        if self.save_importances == 'y':
            plt.savefig('feat_imp_' + self.file_name + '.png')

    def _fit(self, df, y):
        """
        Method trains a K Fold cross validation model on train data set df, performing the following steps for each fold:
        - trains a model on k-1 folds (train_ subset);
        - generates predictions for out of hold fold (val_ subset);
        - adds features importances to self.importances;
        - adds a trained model for to self.mlds;
        - adds a scoring to scores.
        After all K folds are processed, it calculates a final scoring for Kfold model, averaged on per fold scorings
        and bilds self.oof_preds dataframe with the following structure:
       - id;
       - label;
       - predicted label.
        """
        folds = KFold(n_splits=self.fold, shuffle=True, random_state=self.seed)
        self.mdls = []
        self.oof_preds = pd.DataFrame()
        self.importances = pd.DataFrame()
        scores = []

        oof_preds = np.zeros((len(df)))
        for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
            trn_x, trn_y = df.iloc[trn_], y.iloc[trn_]
            val_x, val_y = df.iloc[val_], y.iloc[val_]

            self.mdl.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=self.metrics_dict[self.eval_func]['metric_name'],
                verbose=100,
                early_stopping_rounds=200)

            if self.mdl.objective_ == 'regression':
                oof_preds[val_] = self.mdl.predict(val_x, num_iteration=self.mdl.best_iteration_)
            else:
                oof_preds[val_] = self.mdl.predict_proba(val_x, num_iteration=self.mdl.best_iteration_)

            print('Completed KFOLD: ' + str(fold_))
            imp_df = pd.DataFrame()
            imp_df['feature'] = df.columns
            imp_df['gain'] = self.mdl.feature_importances_
            imp_df['fold'] = fold_ + 1
            self.importances = pd.concat([self.importances, imp_df], axis=0, sort=False)
            self.mdls.append(self.mdl)
            scores.append(self.metrics_dict[self.eval_func]['sklearn_scoring_function'](val_y, oof_preds[val_]))

        # final score
        self.score = np.mean(scores)
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        self.oof_preds = pd.DataFrame(pd.concat([self.train_id, pd.Series(oof_preds), self.train_y], axis=1))
        self.oof_preds.columns = [self.col_id, self.col_target, self.col_target + '_ini']

    def train(self):
        """
        Train method implements the following steps of training process:
        - reads and preprocesses training data (_get_data_trn);
        - trains a model on training data (_fit);
        - outputs features importances and (optionally) saves it to a png file (_save_importances);
        - saves predicted labels for traning data set to a file (save_prediction);
        - registers generated file in a model register file (register_prediction).
        """
        self._get_data_trn()

        self.features = self.train_x.columns

        # Train a model
        self._fit(self.train_x, self.train_y)

        now = datetime.now()
        self.file_name = gen_file_name(now, self.score, self.file_name_prefix)

        # Display and (optionally) store Feature importances
        self._save_importances()
        # Store prediction of train data set (oof)
        save_prediction(self.oof_preds, folder=self.model_type, file_name=self.file_name, reg_folder=self.reg_folder)
        # Register prediction of train data set (oof) in the register
        register_prediction(date=str(now.strftime('%Y%m%d%H%M')), folder=self.model_type, file_name=self.file_name, \
                            featset_name=self.featset_name, score=self.score,
                            reg_folder=self.reg_folder, reg_file_name=self.reg_file_name)

    def predict(self):
        """
        Predict method implements the following steps of prediction process:
        - reads and preprocesses prediction data (_get_data_prd);
        - generates averaged predictions, using set of models self.mlds, trained during K-fold cross validation process;
        - outputs features importances and (optionally) saves it to a png file (_save_importances);
        - saves predicted labels for prediction data set to a file (save_prediction).
        """
        self._get_data_prd()

        # Make predictions
        preds = None
        for mdl in self.mdls:
            if self.mdl.objective_ == 'regression':
                preds_ = mdl.predict(self.prd_x) / len(self.mdls)
            else:
                preds_ = mdl.predict_proba(self.prd_x) / len(self.mdls)

            if preds is None:
                preds = preds_
            else:
                preds += preds_

        preds = pd.DataFrame(pd.concat([self.prd_id, pd.Series(preds)], axis=1))
        preds.columns = [self.col_id, self.col_target]
        save_prediction(preds, folder=self.model_type, file_name=self.file_name,
                        reg_folder=self.reg_folder, prediction_type='prd')
        del preds
        gc.collect()
        
        