import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
import javalang

import keras
from keras import layers

def build_dataset(name, file, label):
    vocab = {}
    in_valid = []
    input_list = []
    label_list = []
    Type1 = ["Invocation", "Class"]
    Type2 = ["Declaration"]
    Type3 = ["Statement", "Clause"]
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
            for Type in [Type1, Type2, Type3]:
                for key_types in Type:
                    if key_types in str(node_type):
                        flag=1
                        if Type!=Type3:
                            try:
                                node_type = node.name
                            except:
                                continue
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

def build_test_dataset(name, file, label, vocab):
    vocab_missing = []
    in_valid = []
    input_list = []
    label_list = []
    Type1 = ["Invocation", "Class"]
    Type2 = ["Declaration"]
    Type3 = ["Statement", "Clause"]
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
            for Type in [Type1, Type2, Type3]:
                for key_types in Type:
                    if key_types in str(node_type):
                        flag=1
                        if Type!=Type3:
                            try:
                                node_type = node.name
                            except:
                                continue
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

def preprocess(input_list, vocab, max_length=500):
    X = np.zeros((len(input_list), max_length))
    for i, x in enumerate(input_list):
        if not x: continue
        if len(x)>max_length:
            x = x[:max_length] 
        X[i][-len(x):] = [vocab[ele]+1 for ele in x]
    return X

def build_model(feature_dim,
                max_len=500,
                lstm_units=128,
                epoch=50,
                batch_size =5,
                pad_key=0,
                nb_classes = 1,
                dense_activate='relu'):
    input1 = layers.Input(shape=(max_len,))
    current_input = layers.Embedding(input_dim=feature_dim, output_dim=lstm_units)(input1)

    lstm_out = layers.LSTM(lstm_units)(current_input)
    lstm_out = layers.Dense(lstm_units, activation=dense_activate)(lstm_out)
    out = layers.Dense(nb_classes, activation='sigmoid', name='main_output')(lstm_out)
    model = keras.models.Model(inputs=[input1], outputs=[out])

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC()])
    #print(model.summary())
    return model

def get_result_1(add1, add2):
    file1 = pd.read_csv(add1)
    file2 = pd.read_csv(add2)

    limited = ["Invocation", "Class", "Declaration", "Statement", "Clause"]
    X_train, y_train, vocab = build_dataset(file1.metric_name, file1.file, file1.label)
    X_test, y_test = build_test_dataset(file2.metric_name, file2.file, file2.label, vocab)

    X_train = preprocess(X_train, vocab)
    X_test = preprocess(X_test, vocab)

    model = build_model(feature_dim = len(vocab)+1)

    model.compile(loss='binary_crossentropy', metrics=["accuracy", keras.metrics.AUC()], optimizer='adam')

    print("Training...")
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min', restore_best_weights=True)
    history = model.fit(X_train, np.array(y_train), epochs=20, batch_size=10, validation_split=0.2, verbose=0, callbacks=[earlyStopping])
    #print("EPOCH:",len(history.history['val_loss']), history.history['val_loss'])

    y_pred = model.predict(X_test, verbose=0).reshape(-1)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    fscore = (2 * precision * recall) / (precision + recall +10**-10)
    ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    y_pred_class = [x>thresholds[ix] for x in y_pred]

    precision, recall, AUC = metrics.precision_score(y_test, y_pred_class), \
                             metrics.recall_score(y_test, y_pred_class),\
                             metrics.roc_auc_score(y_test, y_pred)
                             
    return precision, recall, fscore[ix], AUC

def get_result(add1, add2):
    precision_list = []
    recall_list = []
    AUC_list = []
    fscore_list = []

    for i in range(5):
        precision, recall, fscore, AUC = get_result_1(add1, add2)
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








