import pandas as pd
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from model import MethodNN
from util import *

def get_result_1():
    train_data = pd.read_pickle('parsed_source.pkl').sample(frac=1)
    word2vec = Word2Vec.load("word2vec_node_64").wv
    train_data, val_data = train_test_split(train_data, test_size=0.20, random_state=42)

    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    ENCODE_DIM = 64
    HIDDEN_DIM = 32
    LABELS = 1
    BATCH_SIZE = 13
    USE_GPU = False

    model = MethodNN(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,USE_GPU, embeddings)
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()
    
    best_val_loss = float("inf")
    epochs = 20
    best_model = None
    best_epoch = -1

    def train():
        total_loss = 0.
        permutation = torch.randperm(len(train_data))
        for i in range(0, len(train_data), BATCH_SIZE):
            idx = permutation[i:i+BATCH_SIZE]
            batch_x = train_data['method_seq'].to_numpy()[idx]
            batch_y = train_data['b_label'].to_numpy()[idx]
            optimizer.zero_grad()
            model.batch_size = len(batch_y)
            model.hidden = model.init_hidden()
            output = model(batch_x)
            loss = loss_function(output[0], Variable(torch.FloatTensor(batch_y)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    def evaluate(eval_model, data):
        total_loss = 0.
        start_time = time.time()
        permutation = torch.randperm(len(data))
    
        y_pred = []
        y_true = []
        with torch.no_grad():
            for i in range(0, len(data), BATCH_SIZE):
                idx = permutation[i:i+BATCH_SIZE]
                batch_x = data['method_seq'].to_numpy()[idx]
                batch_y = data['b_label'].to_numpy()[idx]
                model.batch_size = len(batch_y)
                model.hidden = model.init_hidden()
                output = model(batch_x)
                loss = loss_function(output[0], Variable(torch.FloatTensor(batch_y)))
                total_loss += loss
                y_pred.extend(output[0].reshape(-1,))
                y_true.extend(batch_y.reshape(-1,))
        return total_loss, y_pred, y_true

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss,_,_ = evaluate(model, val_data)
        # print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f}| valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time),train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch

    #print("best_epoch", best_epoch)

    test_data = pd.read_pickle('parsed_source_test.pkl').sample(frac=1)
    #print("TESTING")
    test_loss, y_pred, y_true = evaluate(best_model, test_data)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    fscore = (2 * precision * recall) / (precision + recall+10**-10)
    ix = np.argmax(fscore)
    y_pred_class = [x>thresholds[ix] for x in y_pred]

    precision, recall, AUC = metrics.precision_score(y_true, y_pred_class), \
                             metrics.recall_score(y_true, y_pred_class),\
                             metrics.roc_auc_score(y_true, y_pred)
                             
    return precision, recall, fscore[ix], AUC

def get_result(add1, add2):
    precision_list = []
    recall_list = []
    AUC_list = []
    fscore_list = []

    for i in range(5):
        precision, recall, fscore, AUC = get_result_1()
        #print(precision, recall, fscore, AUC)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        AUC_list.append(AUC)

    print("Train:%s, Test:%s" %(add1.split("/")[-1], add2.split("/")[-1]))
    print("Precision=%0.3f, Recall=%0.3f, F-Score=%.3f, AUC=%0.3f\n" % (np.mean(precision_list), \
                                                                        np.mean(recall), \
                                                                        np.mean(fscore_list),\
                                                                        np.mean(AUC_list)))

# CAMEL
address1 = "../../Dataset/AST/camel/camel-1.2.csv"
address2 = "../../Dataset/AST/camel/camel-1.4.csv"
address3 = "../../Dataset/AST/camel/camel-1.6.csv"
get_data(address1, address2)
get_result(address1, address2)
get_data(address2, address3)
get_result(address2, address3)

# # Jedit
# address1 = "../../Dataset/AST/jedit/jedit-3.2.csv"
# address2 = "../../Dataset/AST/jedit/jedit-4.0.csv"
# address3 = "../../Dataset/AST/jedit/jedit-4.1.csv"
# # get_data(address1, address2)
# # get_result(address1, address2)
# # get_data(address2, address3)
# get_result(address2, address3)

# # Log4j
# address1 = "../../Dataset/AST/log4j/log4j-1.0.csv"
# address2 = "../../Dataset/AST/log4j/log4j-1.1.csv"
# get_data(address1, address2)
# get_result(address1, address2)

# # lucene
# address1 = "../../Dataset/AST/lucene/lucene-2.0.csv"
# address2 = "../../Dataset/AST/lucene/lucene-2.2.csv"
# address3 = "../../Dataset/AST/lucene/lucene-2.4.csv"
# get_data(address1, address2)
# get_result(address1, address2)
# get_data(address2, address3)
# get_result(address2, address3)

# # xalan
# address1 = "../../Dataset/AST/xalan/xalan-2.4.csv"
# address2 = "../../Dataset/AST/xalan/xalan-2.5.csv"
# address3 = "../../Dataset/AST/xalan/xalan-2.6.csv"
# get_data(address1, address2)
# get_result(address1, address2)
# get_data(address2, address3)
# get_result(address2, address3)

# # synapse
# address1 = "../../Dataset/AST/synapse/synapse-1.0.csv"
# address2 = "../../Dataset/AST/synapse/synapse-1.1.csv"
# address3 = "../../Dataset/AST/synapse/synapse-1.2.csv"
# get_data(address1, address2)
# get_result(address1, address2)
# get_data(address2, address3)
# get_result(address2, address3)

# # xerces
# address1 = "../../Dataset/AST/xerces/xerces-1.2.csv"
# address2 = "../../Dataset/AST/xerces/xerces-1.3.csv"
# address3 = "../../Dataset/AST/xerces/xerces-1.4.csv"
# # get_data(address1, address2)
# get_result(address1, address2)
# get_data(address2, address3)
# get_result(address2, address3)








