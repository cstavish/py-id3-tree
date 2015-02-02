# ID3 Algorithm
# CS 1675 Assignment #2
# Coleman Stavish

import sys

# define gain functions

def entropy():
	return

def misclassification():
	return

def gini():
	return

class ID3_Tree:
	def __init__(self, labels, features, examples, gain_func):
		self.labels = labels
		self.features = features
		self.examples = examples
		self.gain_func = gain_func

	def _train(self, examples, features):
		root = {}

		# if all examples have the same label
			# return root with that label



		# if features is empty
			# return root with most common label in examples

		# compute gain() for all features
		# associate root with xi* (highest gain)

		# for each possible value of xi*:
			# find Sv (subset of examples where xi*=v)
			# if Sv is not empty:
				# recur with Sv, and all features except xi*
				# make that subtree roots child at branch v (root.values[v] = subtree)
			# else
				# make leaf node (string, as opposed to object) of most common label in examples
				# root.values[v] = leaf, return root

	def train(self):
		return

	def classify(self, object):
		return

def main():
	features = []
	examples = []
	labels = []
	gain_func = None

	if len(sys.argv) < 5:
		print '{0}: arguments required.'.format(sys.argv[0])
		sys.exit(1)

	if sys.argv[1] == 'entropy':
		gain_func = entropy
	elif sys.argv[1] == 'misclassification':
		gain_func = misclassification
	elif sys.argv[1] == 'gini':
		gain_func = gini
	else:
		print '{0}: invalid gain function (must choose `entropy`, `misclassification`, or `gini`.'.format(sys.argv[0])
		sys.exit(1)

	with open(sys.argv[2]) as config_f:
		for idx, line in enumerate(config_f.readlines()):
			# get possible class labels
			if idx == 0:
				labels = line.rstrip().split(',')
			else:
				tokens = line.rstrip().split(',')
				feat = {'name': tokens[0], 'values': tokens[1:]}
				features.append(feat);
		config_f.close()

	with open(sys.argv[3]) as train_f:
		for idx, line in enumerate(train_f.readlines()):
			tokens = line.rstrip().split(',')
			example = {'label': tokens[1]}
			feature_dict = {}
			for idx, val in enumerate(tokens[2:]):
				feature_name = features[idx]['name']
				feature_dict[feature_name] = val
			example['values'] = feature_dict
			examples.append(example)
		train_f.close()

	tree = ID3_Tree(labels, features, examples, gain_func);
	tree.train()



if __name__ == '__main__':
	main()



