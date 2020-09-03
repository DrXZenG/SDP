import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics


def get_result(add1, add2):
    file1 = pd.read_csv(add1)
    file2 = pd.read_csv(add2)

    X_train = file1.iloc[:,3:-1].to_numpy()
    y_train = file1.iloc[:,-1:].to_numpy()
    y_train = np.array([1 if x[0] else 0 for x in y_train])


    X_test = file2.iloc[:,3:-1].to_numpy()
    y_test = file2.iloc[:,-1:].to_numpy()
    y_test = np.array([1 if x[0] else 0 for x in y_test])


    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

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
address1 = "../../Dataset/metric/camel/camel-1.2.csv"
address2 = "../../Dataset/metric/camel/camel-1.4.csv"
address3 = "../../Dataset/metric/camel/camel-1.6.csv"
get_result(address1, address2)
get_result(address2, address3)

# # Jedit
# address1 = "../../Dataset/metric/jedit/jedit-3.2.csv"
# address2 = "../../Dataset/metric/jedit/jedit-4.0.csv"
# address3 = "../../Dataset/metric/jedit/jedit-4.1.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # Log4j
# address1 = "../../Dataset/metric/log4j/log4j-1.0.csv"
# address2 = "../../Dataset/metric/log4j/log4j-1.1.csv"
# address3 = "../../Dataset/metric/log4j/log4j-1.2.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # lucene
# address1 = "../../Dataset/metric/lucene/lucene-2.0.csv"
# address2 = "../../Dataset/metric/lucene/lucene-2.2.csv"
# address3 = "../../Dataset/metric/lucene/lucene-2.4.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # xalan
# address1 = "../../Dataset/metric/xalan/xalan-2.4.csv"
# address2 = "../../Dataset/metric/xalan/xalan-2.5.csv"
# address3 = "../../Dataset/metric/xalan/xalan-2.6.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # synapse
# address1 = "../../Dataset/metric/synapse/synapse-1.0.csv"
# address2 = "../../Dataset/metric/synapse/synapse-1.1.csv"
# address3 = "../../Dataset/metric/synapse/synapse-1.2.csv"
# get_result(address1, address2)
# get_result(address2, address3)

# # xerces
# address1 = "../../Dataset/metric/xerces/xerces-1.2.csv"
# address2 = "../../Dataset/metric/xerces/xerces-1.3.csv"
# address3 = "../../Dataset/metric/xerces/xerces-1.4.csv"
# get_result(address1, address2)
# get_result(address2, address3)








