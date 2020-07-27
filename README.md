# SDP

## Dataset

- df_log4j_v10.csv: contains [java file name, java file content, defective label] for log4j project.
- log4j_v10.csv: CK-metrics-calculated features of the java files in the log4j project. 


## Models

### Metric
- log4j_metric.ipynb: using logistic regression on the CK-metrics-calculated features

### AST: extract certain kinds of the AST nodes.
- MLP_log4j.ipynb: using MLP on the extracted AST nodes
- RNN_log4j.ipynb: using Bidirection-LSTM

### ASTNN paper: http://xuwang.tech/paper/astnn_icse2019.pdf
related files: train.ipynb, process.ipynb, model.py
need packages: javalang, pytorch, gensim
run: 
1. run process.ipynb step by step
2. run train.ipynb step by step
