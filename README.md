# SDP

## Dataset

- metric folder: contains metric for "camel" project, contains CK-metrics-calculated features.
- AST foder: contains [java file name, java file content, defective label] for "camel" project.


## Models

### baseline
- metric.py: using logistic regression on the CK-metrics-calculated features

- RBM.py: using RBM on the extracted AST nodes
- RNN.py: using Bidirection-LSTM 

### astnn 
paper: http://xuwang.tech/paper/astnn_icse2019.pdf


need packages: javalang, pytorch, gensim. All can be installed by

```bash
pip install -r requirements-astnn.txt
```

## run baseline models: 
cd Models/baselines/

1. python3 metrics.py
2. python3 RBM.py
3. python3 RNN.py

## run astnn model:
cd Models/astnn/

1. python3 train.py 
