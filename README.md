# Batch Normalization
Folders:
MNIST ----- Store MNIST database
src   ----- source file
results --- where the results are dumped

Algorithm Documentation:
algo.pdf

Run the *init_conf.py* file will preprocess the MNIST database. What it does is to scale the [0,255] to [0,1]. It also initialize the network parameters, including:
1. Network topology
2. Randomized weights, bias, gammas(only used for the original Batch Normalization)
3. Size of training set and testing set

There are three versions of neural network.
* baseline.py: Baseline code.
* bn_v0.py: Batch Normalization proposed by Ioffe & Szegedy
* bn_v1.py: Simplified Batch Normalization
So far only *baseline.py* take 1 command line argument, a boolen or integer to indicate whether written to mySQL data base is desired. Default is FALSE. SQL will be added to other python files soon.

One needs *cPickle,numpy,scipy,mySQL.connector* to run the code. To run it:
'''
python init_conf.py
python baseline.py 1 or python bn_v0.py or python bn_v1.py
'''
The output is the accuracy and loss of the test set as number of epochs increases

The default setup for network is:
1. layers: [784,100,100,100,10]
2. learning rate: 0.1
3. minibatch size: 60
4. epochs: 50
5. weights/bias/gammas are initialized as random numbers following Gaussian distr.

For Batch Normalization, the inference is done via true population averge rather than moving average. E.g. after each epoch, the network parameters are frozen, feedforward is carried out in order to obtain the means and variance of W*U. Default setting uses 3 epochs and mini-batches of size 60 to obtain the population means and variances. These quantities are then used to infer the test set.

The detailed back-propagation algorithm can be found in the algo.pdf file.
