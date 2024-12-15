#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:12:22 2020

@author: madhu
"""

"""
This script runs the LR classifier for every features and outputs results like:
table 6.3 in Renier's thesis. 
This scripts calculates ROC curve for both TIS and ADS for a list of features 
found inside a directory and even for individually found best features. 
"""


import sys
import numpy as np


from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import binarize, scale, robust_scale
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier




def make_ps(df, n_folds=5):
    """
    :param df: full dataframe
    :param n_folds: number of folds

    :return: list indicating which samples
             belong to which fold
    """

    # Get the list of recordings and labels
    recs = np.array(df.r_IDs.unique())

    labels = np.array([df[df.r_IDs == rec].labels.values[0] for rec in recs])

    # skf = StratifiedKFold(labels, n_folds=n_folds)
    skf = StratifiedKFold(n_splits=n_folds)

    y_ps = np.zeros(shape=(len(df.index),))
    
    fold = 0
    
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
    
    # for train_idx, test_idx in skf:
        test_recs = recs[test_idx]

        ps_test_idx = np.array(df[df['r_IDs'].isin(test_recs)].index)

        y_ps[ps_test_idx] = fold

        fold += 1
    
    return PredefinedSplit(y_ps)


def cv_param_estimation_LR(df, feat_names, CV_SCORER, N_jobs):
    

    def run_gridsearch(X, y, Cs, cv, pen_ratios):
        """
        Tailoring specifically for LRCV

        When using a different classifier, will need to use
        GridSearchCV method and this process would
        be different.
        """
        
        # print('It came here 4 ',)
        # sys.stdout.flush()
        
        if CV_SCORER == 'KAPPA':
            kappa_scorer = make_scorer(cohen_kappa_score)
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring=kappa_scorer)

        elif CV_SCORER == 'AUC':
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc')
            
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = N_jobs)
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=5, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = N_jobs)
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', class_weight='balanced', n_jobs = 20)
            
        else:
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv)
        
        # Scale the data
        # X_scaled = scale(X)
        
        X_r_scaled = robust_scale(X)
        
        # Fit GS with this data
        grid_search.fit(X_r_scaled, y)
        
        gs_opt_params_C = {'C': grid_search.C_[0]}
        
        gs_opt_params_l1_ratio = {'l1_ratio': grid_search.l1_ratio_[0]}
        
        # gs_opt_params = grid_search.best_params_
        
        print (gs_opt_params_C)
        print(gs_opt_params_l1_ratio)
        
        '''
        LogisticRegressionCV.scores_ gives the score for all the folds.
        GridSearchCV.best_score_ gives the best mean score over all the folds.
        '''
        
        print ('Local Max auc_roc:', grid_search.scores_[1].max())  # is wrong
        print ('Max auc_roc:', grid_search.scores_[1].mean(axis=0).max())  # is correct
        
        to_save = 'Max auc_roc:' +  str(grid_search.scores_[1].mean(axis=0).max()) + '\n'
        
        to_save += str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)
        # to_save = str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)
        sys.stdout.flush()
        return gs_opt_params_C, gs_opt_params_l1_ratio, to_save
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # print('It came here 3',)
    # sys.stdout.flush()
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = np.array(df_[feat_names])
    
    LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', n_jobs = N_jobs)
    # LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', class_weight='balanced', n_jobs = 20)
    
    Cs = np.logspace(-5, 5, 10)
    pen_ratios = np.linspace(0, 1, num=6)
    
    # Cs = np.logspace(-7, 7, 30)
    # pen_ratios = np.linspace(0, 1, num=21)
    
    
    LR_params_C, LR_params_l1_ratio, to_save = run_gridsearch(X_val, y_val, Cs, cv=ps, pen_ratios= pen_ratios)
    
    LR_model.set_params(**LR_params_C)
    LR_model.set_params(**LR_params_l1_ratio)

    return LR_model, to_save


def cv_param_estimation_LR_muliclass(df, feat_names, CV_SCORER, N_jobs):
    

    def run_gridsearch(X, y, Cs, cv, pen_ratios):
        """
        Tailoring specifically for LRCV

        When using a different classifier, will need to use
        GridSearchCV method and this process would
        be different.
        """
        
        # print('It came here 4 ',)
        # sys.stdout.flush()
        
        if CV_SCORER == 'KAPPA':
            kappa_scorer = make_scorer(cohen_kappa_score)
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring=kappa_scorer)

        elif CV_SCORER == 'AUC':
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc')
            
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc_ovo_weighted', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = N_jobs)
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc_ovo', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = N_jobs)
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=5, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = N_jobs)
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', class_weight='balanced', n_jobs = 20)
            
        else:
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv)
        
        # Scale the data
        # X_scaled = scale(X)
        
        X_r_scaled = robust_scale(X)
        
        # Fit GS with this data
        grid_search.fit(X_r_scaled, y)
        
        gs_opt_params_C = {'C': grid_search.C_[0]}
        
        gs_opt_params_l1_ratio = {'l1_ratio': grid_search.l1_ratio_[0]}
        
        # gs_opt_params = grid_search.best_params_
        
        print (gs_opt_params_C)
        print(gs_opt_params_l1_ratio)
        
        '''
        LogisticRegressionCV.scores_ gives the score for all the folds.
        GridSearchCV.best_score_ gives the best mean score over all the folds.
        '''
        
        print ('Local Max auc_roc:', grid_search.scores_[1].max())  # is wrong
        print ('Max auc_roc:', grid_search.scores_[1].mean(axis=0).max())  # is correct
        
        to_save = 'Max auc_roc:' +  str(grid_search.scores_[1].mean(axis=0).max()) + '\n'
        
        to_save += str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)
        # to_save = str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)
        sys.stdout.flush()
        return gs_opt_params_C, gs_opt_params_l1_ratio, to_save
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # print('It came here 3',)
    # sys.stdout.flush()
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = np.array(df_[feat_names])
    
    LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', n_jobs = N_jobs)
    # LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', class_weight='balanced', n_jobs = 20)
    
    Cs = np.logspace(-5, 5, 10)
    pen_ratios = np.linspace(0, 1, num=6)
    
    # Cs = np.logspace(-7, 7, 30)
    # pen_ratios = np.linspace(0, 1, num=21)
    
    
    LR_params_C, LR_params_l1_ratio, to_save = run_gridsearch(X_val, y_val, Cs, cv=ps, pen_ratios= pen_ratios)
    
    LR_model.set_params(**LR_params_C)
    LR_model.set_params(**LR_params_l1_ratio)

    return LR_model, to_save







def cv_param_estimation_MLP(df, feat_names, CV_SCORER, N_jobs):
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = np.array(df_[feat_names])
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    
    
    '''
    MLP Classifier
    '''
    # mlp_gs = MLPClassifier(max_iter=10000000, solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(5, 2)) # Run 1.
    mlp_gs = MLPClassifier(max_iter=10000000) # Run 2
    # mlp_gs = MLPClassifier(max_iter=10000000, solver='sgd', learning_rate='adaptive') # Run 3.
    
    
    # define the grid search parameters   
    
    Alpha = np.logspace(-7, 5, 5)
    momentum_list = np.linspace(0, 1, num=5)
    random_state_list = [int(x) for x in np.linspace(1, 10, num=5)]
    
    # activation_list = ['identity', 'logistic', 'tanh', 'relu']
    # solver_list = ['lbfgs', 'sgd', 'adam']
    
    
    # Alpha = [1e-07]
    # momentum_list = [0.1]
    # random_state_list = [1]
    
    param_grid = dict(alpha=Alpha, momentum=momentum_list, random_state=random_state_list)
    # param_grid = dict(alpha=Alpha, momentum=momentum_list, random_state=random_state_list, activation=activation_list, solver=solver_list)
    # param_grid = dict(alpha=Alpha, batch_size=batch_size, momentum=momentum_list, random_state=random_state_list)
    
    grid = GridSearchCV(estimator=mlp_gs, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=1, n_jobs = N_jobs) # use 5 otherwise
    
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')    
    
    mlp_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    return mlp_gs, to_save



def cv_param_estimation_MLP_milticlass(df, feat_names, CV_SCORER, N_jobs):
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = np.array(df_[feat_names])
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    
    
    '''
    MLP Classifier
    '''
    # mlp_gs = MLPClassifier(max_iter=10000000, solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(5, 2)) # Run 1.
    mlp_gs = MLPClassifier(max_iter=10000000) # Run 2
    # mlp_gs = MLPClassifier(max_iter=10000000, solver='sgd', learning_rate='adaptive') # Run 3.
    
    
    # define the grid search parameters   
    
    Alpha = np.logspace(-7, 5, 5)
    momentum_list = np.linspace(0, 1, num=5)
    random_state_list = [int(x) for x in np.linspace(1, 10, num=5)]
    
    # activation_list = ['identity', 'logistic', 'tanh', 'relu']
    # solver_list = ['lbfgs', 'sgd', 'adam']
    
    
    # Alpha = [1e-07]
    # momentum_list = [0.1]
    # random_state_list = [1]
    
    param_grid = dict(alpha=Alpha, momentum=momentum_list, random_state=random_state_list)
    # param_grid = dict(alpha=Alpha, momentum=momentum_list, random_state=random_state_list, activation=activation_list, solver=solver_list)
    # param_grid = dict(alpha=Alpha, batch_size=batch_size, momentum=momentum_list, random_state=random_state_list)
    
    # grid = GridSearchCV(estimator=mlp_gs, param_grid=param_grid, cv=ps, scoring='roc_auc_ovo_weighted', verbose=1, n_jobs = N_jobs) # use 5 otherwise
    grid = GridSearchCV(estimator=mlp_gs, param_grid=param_grid, cv=ps, scoring='roc_auc_ovo', verbose=1, n_jobs = N_jobs) # use 5 otherwise
    
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')    
    
    mlp_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    return mlp_gs, to_save


    

def cv_param_estimation_MLP_TF(df, feat_names, CV_SCORER, N_jobs):
    # def make_ps(df, n_folds=4):
    #     """
    #     :param df: full dataframe
    #     :param n_folds: number of folds

    #     :return: list indicating which samples
    #              belong to which fold
    #     """

    #     # Get the list of recordings and labels
    #     recs = np.array(df.r_IDs.unique())

    #     labels = np.array([df[df.r_IDs == rec].labels.values[0] for rec in recs])

    #     # skf = StratifiedKFold(labels, n_folds=n_folds)
    #     skf = StratifiedKFold(n_splits=n_folds)

    #     y_ps = np.zeros(shape=(len(df.index),))

    #     fold = 0
        
    #     for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
    #     # for train_idx, test_idx in skf:
    #         test_recs = recs[test_idx]

    #         ps_test_idx = np.array(df[df['r_IDs'].isin(test_recs)].index)

    #         y_ps[ps_test_idx] = fold

    #         fold += 1

    #     return PredefinedSplit(y_ps)
    
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = df_[feat_names]
    
    # Function to create model, required for KerasClassifier
    def create_model(dropout_rate=0.0, weight_constraint=0, neurons=1, learn_rate=0.01, momentum=0):
        # create model
        model = Sequential()
        # model.add(Dense(neurons, input_dim=8, activation='relu', kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dense(neurons, input_dim=input_dim, activation='relu', kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(2, activation='softmax'))
        # Compile model
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.compile(loss='mse', optimizer=optimizer, metrics=['AUC'])
        return model
    
    input_dim = X_val.shape[1]
    y_val = y_val.astype(int)
    
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    
    # # define the grid search parameters
    # batch_size = [10, 20, 40, 60, 80, 100]
    # epochs = [10, 50, 100]
    # weight_constraint = [1, 2, 3, 4, 5]
    # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # neurons = [1, 5, 10, 15, 20, 25, 30]
    # learn_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    
    
    # define the grid search parameters
    batch_size = [5, 10, 30]
    dropout_rate = [0.1, 0.2, 0.3]
    epochs = [10, 30, 50]
    learn_rate = [0.001, 0.01, 0.1]
    momentum = [0.1, 0.2, 0.4]
    neurons = [5, 10, 25]
    weight_constraint = [1, 2, 3]
    
    
    # # define the grid search parameters
    # batch_size = [10]
    # epochs = [10]
    # weight_constraint = [1]
    # dropout_rate = [0.1, 0.2]
    # neurons = [5, 10]
    # learn_rate = [0.001, 0.01]
    # momentum = [0.1]
    
    param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate, weight_constraint=weight_constraint, neurons=neurons, learn_rate=learn_rate, momentum=momentum)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=5, n_jobs = N_jobs)
    # grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3)
    
    grid_result = grid.fit(X_val, y_val, verbose=1)
    # summarize results
    print("\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'\n\n')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("\n%f (%f) with: %r" % (mean, stdev, param))
    
    
    model.set_params(**grid_result.best_params_)
    
    to_save = str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)

    return model, to_save



    
def cv_param_estimation_SVM(df, feat_names, CV_SCORER, N_jobs):
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = np.array(df_[feat_names])
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    
    
    '''
    SVC Classifier
    '''
    svc_gs = SVC(probability=True)
    
    # # define the grid search parameters    
    # Cs = [0.1, 1, 10, 100, 1000]
    # gamma_list = [1, 0.1, 0.01, 0.001, 0.0001]
    
    Cs = np.logspace(-7, 7, 30)
    gamma_list = np.logspace(-5, 5, 20)
    # random_state_list = [1, 3, 5, 7, 10]
    # random_state_list = [0.1, 0.3, 0.5, 0.7, 1]
    random_state_list = [1, 10, 20]
    # Cs = [100]
    # gamma_list = [0.001]
    
    param_grid = dict(C=Cs, gamma=gamma_list, random_state=random_state_list)
    # param_grid = dict(alpha=Alpha, batch_size=batch_size, momentum=momentum_list, random_state=random_state_list)
    
    grid = GridSearchCV(estimator=svc_gs, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=1, n_jobs = N_jobs) # use 5 otherwise
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')
    
    
    svc_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    
    return svc_gs, to_save




def cv_param_estimation_SVM_multiclass(df, feat_names, CV_SCORER, N_jobs):
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = np.array(df_[feat_names])
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    
    
    '''
    SVC Classifier
    '''
    svc_gs = SVC(probability=True)
    
    # # define the grid search parameters    
    # Cs = [0.1, 1, 10, 100, 1000]
    # gamma_list = [1, 0.1, 0.01, 0.001, 0.0001]
    
    Cs = np.logspace(-7, 7, 30)
    gamma_list = np.logspace(-5, 5, 20)
    # random_state_list = [1, 3, 5, 7, 10]
    # random_state_list = [0.1, 0.3, 0.5, 0.7, 1]
    random_state_list = [1, 10, 20]
    # Cs = [100]
    # gamma_list = [0.001]
    
    param_grid = dict(C=Cs, gamma=gamma_list, random_state=random_state_list)
    # param_grid = dict(alpha=Alpha, batch_size=batch_size, momentum=momentum_list, random_state=random_state_list)
    
    # grid = GridSearchCV(estimator=svc_gs, param_grid=param_grid, cv=ps, scoring='roc_auc_ovo_weighted', verbose=1, n_jobs = N_jobs) # use 5 otherwise
    grid = GridSearchCV(estimator=svc_gs, param_grid=param_grid, cv=ps, scoring='roc_auc_ovo', verbose=1, n_jobs = N_jobs) # use 5 otherwise
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')
    
    
    svc_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    
    return svc_gs, to_save





def cv_param_estimation_KNN(df, feat_names, CV_SCORER, N_jobs):
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.labels.values
    # data
    X_val = df_[feat_names]
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    # row, column = np.array(X_val).shape
    
    # X_val = list(np.array(X_val).reshape(row, 1, column))
    # y_val = list(np.array(y_val))
    
    
    '''
    KNeighbors Classifier
    '''
    knb_gs = KNeighborsClassifier(n_jobs=N_jobs)
    # knb_gs = KNeighborsClassifier(weights='distance', n_jobs=20)
    
    # # define the grid search parameters    
    n_neighbors_list = [50, 200, 500, 2000]
    leaf_size_list = [2, 5, 10, 30]
    Ps = [1, 2]
    weights_list = ['uniform', 'distance']
    
    param_grid = dict(n_neighbors=n_neighbors_list, leaf_size=leaf_size_list, p=Ps, weights=weights_list)
    
    
    # leaf_size = list(range(1,50))
    # n_neighbors = list(range(1,30))
    # p=[1,2]
    # #Convert to dictionary
    # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    
    # clf = GridSearchCV(knb_gs, hyperparameters, cv=ps, scoring='roc_auc')
    # #Fit the model
    # best_model = clf.fit(X_val, y_val) 
    
    
    
    grid = GridSearchCV(estimator=knb_gs, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=5, n_jobs = N_jobs) # use 5 otherwise
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')
    
       
    
    knb_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    
    return knb_gs, to_save








