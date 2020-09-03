import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import javalang


def build_dataset(name, file, label, limited):
    vocab = {}
    in_valid = []
    input_list = []
    label_list = []
    for i, x in enumerate(file):
        try:
            tree = javalang.parse.parse(x)
        except:
            in_valid.append(name[i])
            continue
        input_ = []
        for path, node in tree:
            node_type = type(node)
            flag =0
            for limit in limited:
                if limit in str(node_type):
                    flag = 1
                    break
            if not flag:continue
            if node_type not in vocab:
                vocab[node_type] = len(vocab)
            input_.append(node_type)
        input_list.append(input_)
        if label[i]>0:
            label_list.append(1)
        else:label_list.append(0)
    return input_list, label_list, vocab

def build_test_dataset(name, file, label, limited, vocab):
    in_valid = []
    vocab_missing = []
    input_list = []
    label_list = []
    for i, x in enumerate(file):
        try:
            tree = javalang.parse.parse(x)
        except:
            in_valid.append(name[i])
            continue
        input_ = []
        for path, node in tree:
            node_type = type(node)
            flag =0
            for limit in limited:
                if limit in str(node_type):
                    flag = 1
                    break
            if not flag:continue
            if node_type not in vocab:
                vocab_missing.append(node_type)
                continue
            input_.append(node_type)
        input_list.append(input_)
        if label[i]>0:
            label_list.append(1)
        else:label_list.append(0)
    return input_list, label_list

def preprocess(input_list, vocab):
    X = np.zeros((len(input_list), len(vocab)))
    for i, x in enumerate(input_list):
        for ele in x:
            X[i][vocab[ele]] =1
    return X

def get_result(add1, add2):
    file1 = pd.read_csv(add1)
    file2 = pd.read_csv(add2)

    limited = ["Invocation", "Class", "Declaration", "Statement", "Clause"]
    X_train, y_train, vocab = build_dataset(file1.metric_name, file1.file, file1.label, limited)
    X_test, y_test = build_test_dataset(file2.metric_name, file2.file, file2.label,limited, vocab)
    X_train = preprocess(X_train, vocab)
    X_test = preprocess(X_test, vocab)

    logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
    rbm = BernoulliRBM(random_state=0, n_iter=50, verbose=0)

    rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    rbm_features_classifier.fit(X_train, y_train)

    y_pred = rbm_features_classifier.predict_proba(X_test)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred[:,1])
    fscore = (2 * precision * recall) / (precision + recall +10**-10)
    ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    y_pred_class = [x>thresholds[ix] for x in y_pred[:,1]]

    precision, recall, AUC = metrics.precision_score(y_test, y_pred_class), \
                             metrics.recall_score(y_test, y_pred_class),\
                             metrics.roc_auc_score(y_test, y_pred[:,1])
                             
    print("Train:%s, Test:%s" %(add1.split("/")[-1], add2.split("/")[-1]))
    print("Precision=%0.3f, Recall=%0.3f, F-Score=%.3f, AUC=%0.3f\n" % (precision, recall, fscore[ix], AUC))
    return precision, recall, fscore[ix], AUC

# CAMEL
address1 = "../../Dataset/AST/camel/camel-1.2.csv"
address2 = "../../Dataset/AST/camel/camel-1.4.csv"
address3 = "../../Dataset/AST/camel/camel-1.6.csv"
get_result(address1, address2)
get_result(address2, address3)

# # Jedit
# address1 = "../../Dataset/AST/jedit/jedit-3.2.csv"
# address2 = "../../Dataset/AST/jedit/jedit-4.0.csv"
# address3 = "../../Dataset/AST/jedit/jedit-4.1.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # Log4j
# address1 = "../../Dataset/AST/log4j/log4j-1.0.csv"
# address2 = "../../Dataset/AST/log4j/log4j-1.1.csv"
# get_result(address1, address2)

# # lucene
# address1 = "../../Dataset/AST/lucene/lucene-2.0.csv"
# address2 = "../../Dataset/AST/lucene/lucene-2.2.csv"
# address3 = "../../Dataset/AST/lucene/lucene-2.4.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # xalan
# address1 = "../../Dataset/AST/xalan/xalan-2.4.csv"
# address2 = "../../Dataset/AST/xalan/xalan-2.5.csv"
# address3 = "../../Dataset/AST/xalan/xalan-2.6.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # synapse
# address1 = "../../Dataset/AST/synapse/synapse-1.0.csv"
# address2 = "../../Dataset/AST/synapse/synapse-1.1.csv"
# address3 = "../../Dataset/AST/synapse/synapse-1.2.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # xerces
# address1 = "../../Dataset/AST/xerces/xerces-1.2.csv"
# address2 = "../../Dataset/AST/xerces/xerces-1.3.csv"
# address3 = "../../Dataset/AST/xerces/xerces-1.4.csv"
# get_result(address1, address2)
# get_result(address2, address3)








