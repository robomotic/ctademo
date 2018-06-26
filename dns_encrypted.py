"""
Slide 19 from:
https://www.slideshare.net/epokh1/the-future-of-threat-intelligence-platforms

This example involves learning using DNS log data from several members of the alliance.

There are the following buckets:
- Alexa domains,
- Goz botnet domains,
- NewGoz botnet domains

The target variable is to predict whether a domain is DGA or not.
This is a binary classification problem so we need a logit implementation.

The DNS' data is split between 3 members of the alliance, all sharing the same features
but different buckets. We refer to this scenario as horizontally partitioned.

The objective is to make use of the whole (virtual) training set to improve
upon the model that can be trained locally by each member of the alliance.

An additional agent is the 'server' who facilitates the information exchange
among the members of the alliance under the following privacy constraints:

1) The individual members's DNS logs are private and cannot be shared even in encrypted form.
2) Information derived (read: gradients) from any members's dataset
   cannot be shared, unless it is first encrypted.
3) None of the parties (members AND server) should be able to infer from
   (which member) a DNS log in the training set has been originated.

Note that we do not protect from inferring IF a particular members's data
has been used during learning. Differential privacy could be used on top of
our protocol for addressing the problem. For simplicity, we do not discuss
it in this example.

In this example linear regression is solved by gradient descent. The server
creates a paillier public/private keypair and does not share the private key.
The member clients are given the public key. The protocol works as follows.
Until convergence: member 1 computes its gradient, encrypts it and sends it
to member 2; member 2 computes its gradient, encrypts and sums it to
member 1's; member 3 does the same and passes the overall sum to the
server. The server obtains the gradient of the whole (virtual) training set;
decrypts it and sends the gradient back - in the clear - to every client.
The clients then update their respective local models.

From the learning viewpoint, notice that we are NOT assuming that each
member sees an unbiased sample from the same DNS' distribution:
this is why the trainign set is biased with particular buckets.
The test set is instead an unbiased sample from the overall distribution.

From the security viewpoint, we consider all parties to be "honest but curious".
Even by seeing the aggregated gradient in the clear, no participant can pinpoint
where members' data originated. This is true if this RING protocol is run by
at least 3 clients, which prevents reconstruction of each others' gradients
by simple difference.

This example was inspired by Google's work on secure protocols for federated
learning[2].

[1]: http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
[2]: https://research.googleblog.com/2017/04/federated-learning-collaborative.html

Dependencies: numpy, sklearn
"""

__author__      = "Paolo Di Prodi"
__email__   = "paolo.research@fortinet.com"

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import phe as paillier
import pandas as pd
from sklearn.linear_model import LogisticRegression
seed = 42
np.random.seed(seed)
import os
import math

ALLOWED_DOMAIN = '0123456789abcdefghijklmnopqrstuvwxyz-'

def compute_domain_entropy(domain):
	stList = list(domain)
	alphabet = list(ALLOWED_DOMAIN)  # list of symbols in the string
	# calculate the frequency of each symbol in the string
	freqList = []
	for symbol in alphabet:
	    ctr = 0
	    for sym in stList:
	        if sym == symbol:
	            ctr += 1
	    freqList.append(float(ctr) / len(stList))

	# Shannon entropy
	ent = 0.0
	for freq in freqList:
		if freq > 0:
			ent = ent + freq * math.log(freq, 2)

	return -ent

def get_data_dga(n_clients):
	pd.set_option('display.max_columns', None)

	if os.path.exists('sampledga.csv') == True:
		dga_pd = pd.read_csv('sampledga.csv',sep=',',quotechar='"',index_col=0)
	else:
		raise Exception("Unable to find test file")

	print(dga_pd.head(10))

	print(dga_pd.tail(10))

	print(dga_pd.describe())

	print(dga_pd.info())

	dga_pd['Entropy(Domain)'] = dga_pd['domain'].apply(compute_domain_entropy)
	dga_pd['Length(Domain)'] = dga_pd['domain'].apply(lambda x : len(x))
	dga_pd['Label'] = dga_pd['class'].apply(lambda x: 1 if x != 'legit' else 0)

	print(dga_pd.head(10))
	print(dga_pd.tail(10))

	print("-----------------")
	print("Unique class {0}".format(dga_pd["class"].unique()))

	print("Unique subclass {0}".format(dga_pd["subclass"].unique()))

	dga_X = dga_pd[['Entropy(Domain)','Length(Domain)']]
	dga_y = dga_pd[['Label']]

	y = dga_pd['Label'].values
	X = dga_X.values
	d = dga_pd['subclass'].values

	for subclass in dga_pd["subclass"].unique():
		print("subclass %s"%subclass)
		filter = np.where(d==subclass)[0]
		print(filter.shape[0])

	X = StandardScaler().fit_transform(X)

	# Add constant to emulate intercept
	X = np.c_[X, np.ones(X.shape[0])]

	# The features are already preprocessed
	# Shuffle
	perm = np.random.permutation(X.shape[0])
	X, y = X[perm, :], y[perm]
	d = d[perm]
	# Select test at random
	test_size = round(X.shape[0]*20/100)

	test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)

	train_idx = np.ones(X.shape[0], dtype=bool)

	train_idx[test_idx] = False

	X_test, y_test = X[test_idx, :], y[test_idx]
	X_train, y_train = X[train_idx, :], y[train_idx]
	d_train = d[train_idx]
	r = ['alexa' 'opendns' 'cryptolocker' 'goz' 'newgoz']
	alexa_idx = np.where( d_train == 'alexa' )
	crypto_idx = np.where(d_train == 'cryptolocker')
	goz_idx = np.where(d_train == 'goz')
	newgoz_idx = np.where(d_train == 'newgoz')
	# create 3 datasets

	# one with alexa + cryptolocker
	# one with alexa + goz
	# The selection is not at random. We simulate the fact that each client
	# sees a potentially very different sample of DNS logs.
	X, y = [], []

	for c in range(n_clients):
		if c == 0:
			X.append(np.concatenate((X_train[alexa_idx],X_train[crypto_idx]),axis=0))
			y.append(np.concatenate((y_train[alexa_idx],y_train[crypto_idx]),axis=0))
		elif c == 1:
			X.append(np.concatenate((X_train[alexa_idx],X_train[goz_idx]),axis=0))
			y.append(np.concatenate((y_train[alexa_idx],y_train[goz_idx]),axis=0))
		elif c == 2:
			X.append(np.concatenate((X_train[alexa_idx],X_train[newgoz_idx]),axis=0))
			y.append(np.concatenate((y_train[alexa_idx],y_train[newgoz_idx]),axis=0))

	return X, y, X_test, y_test

def get_data_artificial(n_clients):
	"""
	Import the dataset via sklearn, shuffle and split train/test.
	Return training, target lists for `n_clients` and a holdout test set
	"""
	print("Loading data")
	X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
	X = StandardScaler().fit_transform(X)
	# add some noise
	rng = np.random.RandomState(2)
	X += 2 * rng.uniform(size=X.shape)

	# Add constant to emulate intercept
	X = np.c_[X, np.ones(X.shape[0])]

	# The features are already preprocessed
	# Shuffle
	perm = np.random.permutation(X.shape[0])
	X, y = X[perm, :], y[perm]

	# Select test at random
	test_size = round(X.shape[0]*20/100)
	test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
	train_idx = np.ones(X.shape[0], dtype=bool)
	train_idx[test_idx] = False
	X_test, y_test = X[test_idx, :], y[test_idx]
	X_train, y_train = X[train_idx, :], y[train_idx]

	# Split train among multiple clients.
	# The selection is not at random. We simulate the fact that each client
	# sees a potentially very different sample of patients.
	X, y = [], []
	step = int(X_train.shape[0] / n_clients)
	for c in range(n_clients):
		X.append(X_train[step * c: step * (c + 1), :])
		y.append(y_train[step * c: step * (c + 1)])

	return X, y, X_test, y_test


def mean_square_error(y_pred, y):
	""" 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
	return np.mean((y - y_pred) ** 2)


def encrypt_vector(public_key, x):
	return [public_key.encrypt(i) for i in x]


def decrypt_vector(private_key, x):
	return np.array([private_key.decrypt(i) for i in x])


def sum_encrypted_vectors(x, y):
	if len(x) != len(y):
		raise ValueError('Encrypted vectors must have the same size')
	return [x[i] + y[i] for i in range(len(x))]


class Server:
	"""Private key holder. Decrypts the average gradient"""

	def __init__(self, key_length):
		 keypair = paillier.generate_paillier_keypair(n_length=key_length)
		 self.pubkey, self.privkey = keypair

	def decrypt_aggregate(self, input_model, n_clients):
		return decrypt_vector(self.privkey, input_model) / n_clients

def sigmoid(scores):
	return 1 / (1 + np.exp(-scores))


class Client:
	"""Runs linear regression with local data or by gradient steps,
	where gradient can be passed in.

	Using public key can encrypt locally computed gradients.
	"""

	def __init__(self, name, X, y, pubkey):
		self.name = name
		self.pubkey = pubkey
		self.X, self.y = X, y
		self.weights = np.zeros(X.shape[1])

	def fit(self, n_iter, eta=0.01):
		"""Linear regression for n_iter"""
		for _ in range(n_iter):
			gradient = self.compute_gradient()
			self.gradient_step(gradient, eta)

	def gradient_step(self, gradient, eta=0.01):
		"""Update the model with the given gradient"""
		self.weights -= eta * gradient

	def compute_gradient(self):
		"""Compute the gradient of the current model using the training set
		"""
		delta = self.predict(self.X) - self.y
		return delta.dot(self.X)

	def predict(self, X):
		"""Score test data"""
		scores = X.dot(self.weights)
		return sigmoid(scores)

	def encrypted_gradient(self, sum_to=None):
		"""Compute and encrypt gradient.

		When `sum_to` is given, sum the encrypted gradient to it, assumed
		to be another vector of the same size
		"""
		gradient = self.compute_gradient()
		encrypted_gradient = encrypt_vector(self.pubkey, gradient)

		if sum_to is not None:
			return sum_encrypted_vectors(sum_to, encrypted_gradient)
		else:
			return encrypted_gradient
import time

def federated_learning(n_iter, eta, n_clients, key_length):
	names = ['CTA Member {}'.format(i) for i in range(1, n_clients + 1)]

	X, y, X_test, y_test = get_data_dga(n_clients=n_clients)


	start = time.time()

	# Instantiate the server and generate private and public keys
	# NOTE: using smaller keys sizes wouldn't be cryptographically safe
	server = Server(key_length=key_length)

	# Instantiate the clients.
	# Each client gets the public key at creation and its own local dataset
	clients = []
	for i in range(n_clients):
		clients.append(Client(names[i], X[i], y[i], server.pubkey))

	# Each client trains a linear regressor on its own data
	print('Accuracy that each client gets on test set by '
		  'training only on own local data:')
	for c in clients:
		c.fit(n_iter, eta)
		y_pred = c.predict(X_test)
		acc = accuracy_score(y_pred.round(), y_test)
		print('{:s}:\t{:.2f}'.format(c.name, acc))

		#clf = LogisticRegression()
		#clf.fit(c.X,c.y)
		#y_pred = c.predict(X_test)
		#acc = accuracy_score(y_pred.round(), y_test)
		#print('Vanilla Logistic Regression {:s}:\t{:.2f}'.format(c.name, acc))

	# The federated learning with gradient descent
	print('Running distributed gradient aggregation for {:d} iterations'
		  .format(n_iter))
	for i in range(n_iter):

		# Compute gradients, encrypt and aggregate
		encrypt_aggr = clients[0].encrypted_gradient(sum_to=None)
		for c in clients:
			encrypt_aggr = c.encrypted_gradient(sum_to=encrypt_aggr)

		# Send aggregate to server and decrypt it
		aggr = server.decrypt_aggregate(encrypt_aggr, n_clients)

		# Take gradient steps
		for c in clients:
			c.gradient_step(aggr, eta)

	print('Accuracy that each client gets after running the protocol:')
	for c in clients:
		y_pred = c.predict(X_test)
		acc = accuracy_score(y_pred.round(), y_test)
		print('{:s}:\t{:.2f}'.format(c.name, acc))

	end = time.time()
	print(end - start)

def federated_learning_test(n_iter, eta, n_clients, key_length):
	names = ['CTA Member {}'.format(i) for i in range(1, n_clients + 1)]

	X, y, X_test, y_test = get_data_artificial(n_clients=n_clients)

	# Instantiate the server and generate private and public keys
	# NOTE: using smaller keys sizes wouldn't be cryptographically safe
	server = Server(key_length=key_length)

	# Instantiate the clients.
	# Each client gets the public key at creation and its own local dataset
	clients = []
	for i in range(n_clients):
		clients.append(Client(names[i], X[i], y[i], server.pubkey))

	# Each client trains a linear regressor on its own data
	print('Accuracy that each client gets on test set by '
		  'training only on own local data:')
	for c in clients:
		c.fit(n_iter, eta)
		y_pred = c.predict(X_test)
		acc = accuracy_score(y_pred.round(), y_test)
		print('{:s}:\t{:.2f}'.format(c.name, acc))

		#clf = LogisticRegression()
		#clf.fit(c.X,c.y)
		#y_pred = c.predict(X_test)
		#acc = accuracy_score(y_pred.round(), y_test)
		#print('Vanilla Logistic Regression {:s}:\t{:.2f}'.format(c.name, acc))

	# The federated learning with gradient descent
	print('Running distributed gradient aggregation for {:d} iterations'
		  .format(n_iter))
	for i in range(n_iter):

		# Compute gradients, encrypt and aggregate
		encrypt_aggr = clients[0].encrypted_gradient(sum_to=None)
		for c in clients:
			encrypt_aggr = c.encrypted_gradient(sum_to=encrypt_aggr)

		# Send aggregate to server and decrypt it
		# aggr = server.decrypt_aggregate(encrypt_aggr, n_clients)

		aggr = np.divide(encrypt_aggr,n_clients)

		# Take gradient steps
		for c in clients:
			c.gradient_step(aggr, eta)

	print('Accuracy that each client gets after running the protocol:')
	for c in clients:
		y_pred = c.predict(X_test)
		acc = accuracy_score(y_pred.round(), y_test)
		print('{:s}:\t{:.2f}'.format(c.name, acc))

import sys
if __name__ == '__main__':
	# Set learning, data split, and security params
	federated_learning(n_iter=50, eta=0.01, n_clients=3, key_length=1024)
