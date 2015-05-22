# Batch Normalization
Folders:
* MNIST ----- Store MNIST database
* src   ----- source file
* results --- where the results are dumped

Algorithm Documentation:
algo.pdf

The detailed back-propagation algorithm can be found in the algo.pdf file.

The *init_conf.py* file preprocesses the MNIST database. What it does is to scale the [0,255] to [0,1]. It also initialize the network parameters, including:
1. Network topology
2. Randomized weights, bias, gammas(only used for the original Batch Normalization)
3. Size of training set and testing set

There are three versions of neural network.
* baseline.py: Baseline code.
* bn_v0.py: Batch Normalization proposed by Ioffe & Szegedy.
In this code, the inference is done via true population averge rather than moving average. E.g. after each epoch, the network parameters are frozen, feedforward is carried out in order to obtain the means and variance of WU. Default setting uses 3 epochs and mini-batches of size 60 to obtain the population means and variances. These quantities are then used to infer the test set.
* bn_v1.py: Simplified Batch Normalization
The simplified BN only centers X=WU. It doesn't divide X by the standard deivation of WU.
So far only *baseline.py* take 1 command line argument to indicate whether written to mySQL data base is desired: 0 (no SQL) or 1 (SQL) Default is FALSE. SQL will be added to other python files soon.

One needs *cPickle,numpy,scipy,mySQL.connector* to run the code. To run it:
```
python init_conf.py
python baseline.py 1 # baseline or
python bn_v0.py # original BN or
python bn_v1.py # simplified BN 
```
The output is the accuracy and loss of the test set as number of epochs increases

The default setup for network is:
* layers: [784,100,100,100,10]
* learning rate: 0.1
* minibatch size: 60
* epochs: 50
* weights/bias/gammas are initialized as random numbers following Gaussian distr.




