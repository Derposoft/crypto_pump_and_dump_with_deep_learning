## Crypto Pump and Dump Detection via Deep Learning
### by Viswanath Chadalapaka, Kyle Chang, Gireesh Mahajan, and Anuj Vasil

Our work shows that deep learning can be applied to cryptocurrency pump and dump (P&D) data to create higher-scoring models than previously found by La Morgia et al., 2020.

To run our code, use:
```
python train.py [--option value]
```

Possible command line options are as follows, by category:

#### *General Settings*:
1. model: Choose between CLSTM, AnomalyTransformer, TransformerTimeSeries, AnomalyTransfomerIntermediate, and AnomalyTransfomerBasic
    1. The TransformerTimeSeries makes use of a standard transformer encoder to establish a baseline w/o anomaly attention
    2. The AnomalyTransfomerBasic is the simplest use of anomaly attention
    3. AnomalyTransfomerIntermediate uses association scores from anomaly attention in the loss, but does not use the minimax optimization strategy. Instead it makes use of just the maximize phase. This is an intermediate between the final AnomalyTransformer model that produces near identical results, but trains much quicker.
    4. This is the finaly AnomalyTransformer that uses the minimax optimziation strategy. Unlike the other models it is much slower to train, but does produce optimal results. 
2. n_epochs: Number of epochs to train the given model

#### *CLSTM Settings*:
1. embedding_size: The embedding size of the CLSTM
2. n_layers: Number of LSTM layers
3. kernel_size: Size of the convolutional kernel
4. dropout: Dropout added to the LSTM layers
5. cell_norm: True/False -- whether or not to normalize the gate values at the LSTM cell level
6. out_norm: True/False -- whether or not to normalize the output of each LSTM layer

#### *Transformer Settings*:
1. feature_size: amount of features to use from the original data
2. n_layers: number of Transformer layers
3. n_head: number of heads in multi-head self attention. Only required for base `TransformerTimeSeries` model
4. lambda_: weight of kl divergences between associations in anomaly attention module. Only required for `AnomalyTransfomerIntermediate` and `AnomalyTransformer`

#### *Training Settings*:
1. lr: Learning rate
2. lr_decay_step: Number of epochs to wait before decaying the learning rate, 0 for no decay
3. lr_decay_factor: Multiplicative learning rate decay factor
4. weight_decay: Weight decay regularization
5. batch_size: Batch size
6. train_ratio: Ratio of data to use for train
7. undersample_ratio: Undersample proportion of majority class
8. segment_length: Length of each segment

#### *Validation Settings*:
1. prthreshold: Set the precision-recall threshold of the model
2. kfolds: Enable a k-fold validation scheme. If set to anything other than 1, train_ratio will be ignored

#### *Ease of Use Settings*:
1. save: Cache processed data for faster experiment startup times
2. validate_every_n: Skips validation every epoch and only validates every n epochs, saves on time
3. train_output_every_n: Doesn't output train loss details for cleaner logs
4. time_epochs: Adds epoch timing to logs
5. final_run: Automatically sets validate_every_n=1 and train_output_every_n=1, used for reproducing paper reults
6. verbose: Additional debug output (average output at 0/1 ground truth labels, etc)
7. dataset: Point to the time-series dataset to train the model on
8. seed: Set the seed of the model
9. run_count: Set the number of times to run the model, in order to compute confidence intervals from logs
