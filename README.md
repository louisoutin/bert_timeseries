# bert_timeseries
Multivariate time series representation learning (using bert-like model)

This is a PyTorch implementation of [A TRANSFORMER-BASED FRAMEWORK FOR MULTIVARIATE TIME SERIES REPRESENTATION LEARNING](https://arxiv.org/pdf/2010.02803.pdf).
Network architecture is taken from https://github.com/timeseriesAI/tsai. Their implementation uses fastai layer on top of pytorch while the current repo use lightning as a training framework. The goal of this repository is to focus only on the unsupervised training mode of the model.
