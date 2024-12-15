#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:31:15 2024

@author: madhupahar
"""

'''
I created this script to do classification using foundation models ... 
I need to finally apply this to the 
'''




import os, sys, glob

# from main import *
# from config import *
# check_env('ACONDA')

from datetime import datetime

from tqdm import tqdm
# from natsort import natsorted

# from multiprocessing.pool import ThreadPool, Pool
# from multiprocessing import cpu_count
import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, RobertaTokenizerFast, RobertaForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification, BartTokenizerFast, BartForSequenceClassification


# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd

# import nltk
# nltk.download("stopwords")

import config_classifiers
# from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt





def read_file(file):
    f = open(file, "r")
    return f.read()




def calc_metrics(actual_labels, pred_vals, avg=None):
    if avg == None: 
        f1_val = f1_score(actual_labels, pred_vals)
        pres_val = precision_score(actual_labels, pred_vals)
        rec_val = recall_score(actual_labels, pred_vals)
    else:
        f1_val = f1_score(actual_labels, pred_vals, average=avg)
        pres_val = precision_score(actual_labels, pred_vals, average=avg)
        rec_val = recall_score(actual_labels, pred_vals, average=avg)
    
    conf_val = confusion_matrix(actual_labels, pred_vals)
    
    return f1_val, pres_val, rec_val, conf_val


    

'''
The following function should calcualte and print the predicted labels based on majority voting. 
The pandas dataframe df should have two columns: r_IDs and pred_label
'''
def majority_voting_pred_labels(df, a_class_type, verbose):
    data = {'r_IDs':df[['r_IDs', 'pred_label']].groupby('r_IDs').mean().index.values, 
            'grouped_pred_label':df[['r_IDs', 'pred_label']].groupby('r_IDs').mean().pred_label.values}
    df_final_test_grouped = df.merge(pd.DataFrame(data), on='r_IDs', how='inner')
    
    # print('df_final_test_grouped before : ')
    # print(df_final_test_grouped)
    
    
    df_final_test_grouped = df_final_test_grouped.drop_duplicates(subset="r_IDs", keep='first')
    
    # print('df_final_test_grouped after : ')
    # print(df_final_test_grouped)
    
    if a_class_type == '3-way':
        thrs_val = 0.333333
        
        final_pred_label_list = []
        for x in df_final_test_grouped.grouped_pred_label:
            if x < thrs_val:
                final_pred_label_list.append(0)
            elif x >= thrs_val and x < (2*thrs_val):
                final_pred_label_list.append(1)
            else:
                final_pred_label_list.append(2)
        
    else:
        thrs_val = 0.5
        
        final_pred_label_list = []
        for x in df_final_test_grouped.grouped_pred_label:
            if x < thrs_val:
                final_pred_label_list.append(0)
            else:
                final_pred_label_list.append(1)
            
    
    
    df_final_test_grouped.insert(len(df_final_test_grouped.columns), 'final_pred_label', final_pred_label_list)
    
    
    
    
    actual_labels = df_final_test_grouped.labels
    
    ## Calculate the initial results 
    col_2_cons = 'final_pred_label'
    
    pred_vals = df_final_test_grouped[col_2_cons]
    
    f1_val, pres_val, rec_val, conf_val = calc_metrics(actual_labels, pred_vals, avg='macro')
    
    # calc_metrics(actual_labels, pred_vals, avg='micro')
    # calc_metrics(actual_labels, pred_vals, avg='weighted')
    
    # print('with threshold = 0.5')
    if verbose == 1:
        print('Macro F1-score: ', round(f1_val, 2))
        print('Macro Precision: ', round(pres_val, 2))
        print('Macro Recall: ', round(rec_val, 2))
        print('conf mat')
        print(conf_val)
    
    if a_class_type == '3-way':
    
        precision, recall, fscore, support = score(actual_labels, pred_vals)
        if verbose == 1:
            print('Metric \t HC \t MCI \t Demen')
            # print('------ \t ------ \t ------ \t ------')
            print('Precis \t {} \t {} \t {}'.format( round(precision[0], 2), round(precision[1], 2), round(precision[2], 2) ))
            print('Recall \t {} \t {} \t {}'.format( round(recall[0], 2), round(recall[1], 2), round(recall[2], 2) ))
            print('F1-val \t {} \t {} \t {}'.format( round(fscore[0], 2), round(fscore[1], 2), round(fscore[2], 2) ))
    
    return f1_val, pres_val, rec_val, conf_val




class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}



        
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    predictions = []
    actual_labels = []
    lossess = []
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # print('attention_mask : ', attention_mask)
        
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # loss = nn.CrossEntropyLoss()(outputs, labels)
        # loss = nn.CrossEntropyLoss(weight=weight_tensor)(outputs, labels).to(device)
        loss = nn.CrossEntropyLoss(weight=weight_tensor)(outputs.logits, labels).to(device)
        # print('loss : ', loss)
        loss.backward()
        # print('loss after backward : ', loss)
        sys.stdout.flush()
        lossess.append(loss.cpu().tolist())
        
        optimizer.step()
        scheduler.step()
        # print('train labels : ', labels)
        # print('train outputs : ', outputs)
        # _, preds = torch.max(outputs, dim=1)
        _, preds = torch.max(outputs.logits, dim=1)
        # print('train preds : ', preds)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        # print('train probs : ', probs)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    # return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
    
    # print('actual_labels : ', actual_labels)
    # print('predictions : ', predictions)
    # sys.stdout.flush()
    # time.sleep(100000000)
    
    return f1_score(actual_labels, predictions), precision_score(actual_labels, predictions), recall_score(actual_labels, predictions), confusion_matrix(actual_labels, predictions), np.mean(lossess), np.std(lossess)



def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    lossess = []
    probabs = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            lossess.append(loss.cpu().tolist())
            
            # print('val labels : ', labels)
            # print('val outputs : ', outputs)
            _, preds = torch.max(outputs.logits, dim=1)
            # print('val preds : ', preds)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            # print('val probs : ', probs)
            predictions.extend(preds.cpu().tolist())
            probabs.extend(probs.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    # return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
    return f1_score(actual_labels, predictions), precision_score(actual_labels, predictions), recall_score(actual_labels, predictions), confusion_matrix(actual_labels, predictions), np.mean(lossess), np.std(lossess), predictions, probabs





def calc_class_weight(train_y):
    """
    Compute class weight given imbalanced training data
    Usually used in the neural network model to augment the loss function (weighted loss function)
    Favouring/giving more weights to the rare classes.
    """
    
    class_list = list(set(train_y))
    # class_weight_value = scikit_class_weight.compute_class_weight(class_weight ='balanced', classes = class_list, y = train_y)
    class_weight_value = compute_class_weight(class_weight ='balanced', classes = np.unique(train_y), y = train_y)
    class_weight = dict()

    # Initialize all classes in the dictionary with weight 1
    curr_max = int(np.max(class_list))
    for i in range(curr_max):
        class_weight[i] = 1

    # Build the dictionary using the weight obtained the scikit function
    for i in range(len(class_list)):
        class_weight[class_list[i]] = class_weight_value[i]

    return class_weight



def get_os_cmd(cuda_ids):
    os_cmd = 'os.environ["CUDA_VISIBLE_DEVICES"] ="'
    
    # os.environ['CUDA_VISIBLE_DEVICES'] ='1,2'
    
    # print('Len of cuda_ids ', len(cuda_ids))
    
    for i in range(len(cuda_ids)):
        if i == 0:
            os_cmd += str(cuda_ids[i])
        else:
            os_cmd += ','+str(cuda_ids[i])
    
    os_cmd += '"'
    
    return os_cmd




if __name__=='__main__' and '__file__' in globals():
    
    
    if len(sys.argv) < 2:
        print('Please use : python CognoSpeak_LLM.py test [where test is the name of the test]')
        sys.exit()
    
    
    
    startTime = datetime.now()
    current_time = startTime.strftime("%Y/%m/%d at %H:%M:%S")
    print('\n\nThe script starting at: ' + str(current_time), ' \n\n' )
    
    
    
    
    given_token = sys.argv[1]
    
    cuda_ids = sys.argv[2].split(',')
    
    
    
    if not type(cuda_ids) is list:
        print("Please input cuda ids as a list like: 0 or 1,3 etc. and make sure all the GPU IDs are correct.")
        sys.exit()
    
    
    # os.environ['CUDA_VISIBLE_DEVICES'] ='1,2'
    cuda_ids.sort()
    os_cmd = get_os_cmd(cuda_ids)
    
    try: 
        exec(os_cmd)
    except:
        print("Please input cuda ids as a list like: 0 or 1,3 etc. and make sure all the GPU IDs are correct.")
        print('Setting CUDA_VISIBLE_DEVICES environment command : ', os_cmd)
        sys.exit()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    
    BASE_DIR = '..'
    
    data_path = BASE_DIR+'/data/CognoSpeak_results/data/'
    feat_path = BASE_DIR+'/data/CognoSpeak_results/feats/'
    results_path = BASE_DIR+'/data/CognoSpeak_results/results/'
    
    
    N_FOLDS = config_classifiers.N_FOLDS
    
    VERBOSE = 0
    
    
    
    
    l_q_types = ['Q4', 'Q6', 'Q10', 'Q12', 'ALL'] 
    
    
    
    
    df_meta_final = pd.read_csv( BASE_DIR+'/data/CognoSpeak_results/metadata.csv' )
    
    
    
    df_feat_name = BASE_DIR+'/data/CognoSpeak_results/feats/CognoSpeak_TEXT.csv'
    
    if os.path.exists(df_feat_name):
        df_feat = pd.read_csv( df_feat_name )
    else: 
        df_feat = pd.DataFrame([], columns=['dir_name', 'Q_type', 'text_info'])
        
        df_meta_final = df_meta_final.reset_index(drop=True)
        for ind in tqdm(df_meta_final.index):
            
            for q_type in l_q_types:
                # a_txt = glob.glob( data_path+df_process_metadata['anyon_IDs'][ind]+'/*__'+ q_type+   '*.txt' )[0]
                a_txt = glob.glob( data_path+df_meta_final['dir_name'][ind]+'/'+df_meta_final['dir_name'][ind]+'*_'+ q_type+   '*.txt' )[0]
                
                text_info = ''
                for a_part in read_file( a_txt ).split('\n'):
                    
                    if a_part != ' ' and a_part != '':
                        txt_speech = a_part.split('\t')[-1].replace('Pat:', '').replace('(Buzzer sounds)', '').strip()
                        text_info += txt_speech
                
               
                
                df_feat.loc[len(df_feat.index)] = [df_meta_final['dir_name'][ind]] + [q_type] + [text_info]
                
                
        
        df_feat = df_feat.merge(df_meta_final, on='dir_name', how='inner')
        # df_feat = df_feat.rename(columns={'anyon_IDs': 'r_IDs'})
        df_feat.to_csv(df_feat_name, index=False)
    
    
    
    num_classes = 2
    max_length = 512
    num_epochs = 30
    
    batch_size = 64
    
    
    
    
    
    l_classes = ['BART', 'DistilBERT', 'RoBERTa']
    
    
    
    l_LR = [0.0001, 0.00005, 0.00001, 0.000005]
    
    
        
    for class_name in l_classes:
        
        for learning_rate in l_LR: 
        
            for Q_2_consider in l_q_types:
                
                print('\n----------------------')
                
                print('Classifier is : ', class_name)
                
                print('Learning rate is: ', learning_rate)
                
                print('Question selected is: ', Q_2_consider)
                
                
                FINAL_TOKEN = given_token + '_' +Q_2_consider + '_' + class_name + '_' + str(max_length) + '_' + str(batch_size) + '_' + str(learning_rate)
                
                list_f1vals = []
                list_pres = []
                list_recalls = []
                
                for k in range(N_FOLDS):
                    
                    if Q_2_consider == 'ALL':
                        df_train = df_feat[df_feat['FOLD_'+str(k)]=='TRAIN' ]
                        df_test = df_feat[df_feat['FOLD_'+str(k)]=='TEST']
                    else:
                        df_train = df_feat[ (df_feat['FOLD_'+str(k)]=='TRAIN') & (df_feat.Q_type==Q_2_consider) ]
                        df_test = df_feat[ (df_feat['FOLD_'+str(k)]=='TEST') & (df_feat.Q_type==Q_2_consider) ]
                
                    
                    
                
                
                    train_texts = list(df_train.text_info)
                    train_labels = [int(y) for y in list(df_train.labels)]
                    
                    
                    val_texts = list(df_test.text_info)
                    val_labels = [int(y) for y in list(df_test.labels)]
                    
                    
                    
                
                
                    ##Sort out the data imbalance ... 
                    
                    weights = calc_class_weight(train_labels)
                    
                    
                    weight_list = []
                    for key, weight in weights.items():
                        weight_list.append(weight)
                    weight_tensor = torch.FloatTensor(weight_list).to(device)
                    
                    
                    
                    if class_name == 'BART':
                        
                        model_id = "facebook/bart-base"
                        tokenizer = BartTokenizerFast.from_pretrained(model_id)
                        model = BartForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
                        
                        
                    elif class_name == 'RoBERTa':
                        model_id = "roberta-base"
                        # model_id = "roberta-large"
                        tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
                        model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
                        
                    elif class_name == 'DistilBERT':
                        model_id = 'distilbert-base-uncased'
                        tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
                        model = DistilBertForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
                        
                        
                        
                        
                    model = nn.DataParallel(model)
                    model.to(device)
                    
                    
                    
                    
                    
                    
                    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
                    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
                    
                    
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                    total_steps = len(train_dataloader) * num_epochs
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
                    
                    
                    train_losses_mean = []
                    train_losses_std = []
                    
                    val_losses_mean = []
                    val_losses_std = []
                    
                    
                    
                    for epoch in range(num_epochs):
                        if VERBOSE == 1 or VERBOSE == 2:
                            print(f"Epoch {epoch + 1}/{num_epochs}")
                        
                        
                        f1_val, pres_val, recall_val, report, m_loss, s_loss = train(model, train_dataloader, optimizer, scheduler, device)
                        
                        if VERBOSE == 2: 
                            print('Train: F1-Score {:.2f}; Presicion {:.2f}; Recall {:.2f}; Mean loss {:.2f}; STD loss {:.2f}'.format(f1_val, pres_val, recall_val, m_loss, s_loss))
                            # print(con_mat)
                        elif VERBOSE == 1:
                            print('Mean Train loss {:.2f}; with STD {:.2f}'.format(m_loss, s_loss))
                        sys.stdout.flush()
                        
                        train_losses_mean.append( m_loss )
                        train_losses_std.append( s_loss )
                        
                        
                        
                        f1_val, pres_val, recall_val, report, m_loss, s_loss, preds, probs = evaluate(model, val_dataloader, device)
                        
                        
                        if VERBOSE == 2: 
                            
                            print('Test: F1-Score {:.2f}; Presicion {:.2f}; Recall {:.2f}; Mean loss {:.2f}; STD loss {:.2f}'.format(f1_val, pres_val, recall_val, m_loss, s_loss))
                            # print(con_mat)
                            print('----------\n\n')
                        elif VERBOSE == 1:
                            print('Mean Test loss {:.2f}; with STD {:.2f}'.format( m_loss, s_loss))
                        sys.stdout.flush()
                        
                        
                        val_losses_mean.append( m_loss )
                        val_losses_std.append( s_loss )
                        
                    
                    # torch.save(model.state_dict(), results_path+FINAL_TOKEN+'__FOLD-'+str(k)+'.save')
                    
                    
                    df_final_test = df_test[['dir_name', 'labels']]
                    
                    df_final_test.insert( len(df_final_test.columns), 'pred_label', preds )
                    
                    df_final_test = df_final_test.rename(columns={'dir_name': 'r_IDs'})
                    
                    ## calculate the majority voting preds and print out the scores ... 
                    f1_val, pres_val, rec_val, conf_val = majority_voting_pred_labels(df_final_test, '2-way', 0)
                    
                    list_f1vals.append( f1_val )
                    list_pres.append( pres_val )
                    list_recalls.append( rec_val )
                
                print('List of F1-scores: ', list_f1vals)
                print('Max F1-score: ', max(list_f1vals))
                print('Mean F1-score: ', round(np.mean(list_f1vals), 2), ' with STD : ', round(np.std(list_f1vals), 2))
                print('Mean Precision: ', round(np.mean(list_pres), 2), ' with STD : ', round(np.std(list_pres), 2))
                print('Mean Recall: ', round(np.mean(list_recalls), 2), ' with STD : ', round(np.std(list_recalls), 2))
                    
                print('----------------------')
                sys.stdout.flush()
                    
                
                
    
    
    executionTime = (datetime.now() - startTime)
    current_time = datetime.now().strftime("%Y/%m/%d at %H:%M:%S")
    print('\n\nThe script completed at: ' + str(current_time))
    print('Execution time: ' + str(executionTime), ' \n\n')


