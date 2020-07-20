# SDP

## Dataset

- df_log4j_v10.csv: contains [java file name, java file content, defective label] for log4j project.
- log4j_v10.csv: CK-metrics-calculated features of the java files in the log4j project. 


## Models

### Metric
- log4j_metric.ipynb: using logistic regression for SDP using CK-metrics-calculated features

### AST: extract certain kinds of the AST nodes.
- MLP_log4j.ipynb: using MLP 
- RNN_log4j.ipynb: using Bidirection-LSTM
