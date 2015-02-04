# ID3 Algorithm
# CS 1675 Assignment #2
# Coleman Stavish

import sys
import math
from collections import Counter

# define gain functions

def entropy(S):
	cnt = Counter([e['label'] for e in S])
	p_labels = [float(cnt[label]) / len(S) for label in cnt.keys()]
	return sum([-p_label * math.log(p_label, 2) for p_label in p_labels])

def misclassification(S):
	cnt = Counter([e['label'] for e in S])
	most_common = cnt.most_common(1)[0]
	return 1 - float(most_common[1]) / len(S)

def gini(S):
	cnt = Counter([e['label'] for e in S])
	return 1 - sum([(float(cnt[label] / len(S))**2) for label in cnt.keys()])

class ID3_Tree:
	def __init__(self, labels, features, examples, diversity_func):
		self.labels = labels
		self.features = features
		self.examples = examples
		self.diversity_func = diversity_func

	def train(self):
		self.root = self._train(self.examples, self.features)

	def print_tree(self):
		self._print_tree(self.root, 0)

	def classify(self, data_point):
		return self._classify(data_point, self.root)

	def _gain(self, feature, examples):
		entropy = self.diversity_func(examples)
		diversity_sum = 0
		for v in feature['values']:
			Sv = []
			for e in examples:
				if e['values'][feature['name']] == v:
					Sv.append(e)
			if len(Sv) > 0:
				diversity_sum += self.diversity_func(Sv) * (float(len(Sv))/len(examples))
		return entropy - diversity_sum

	def _train(self, examples, features):
		root = {}

		# if all examples have the same label
			# return root as leaf node with that label
		all_identical = True
		for i in xrange(1, len(examples)):
			if examples[0]['label'] != examples[i]['label']:
				all_identical = False
				break

		if all_identical:
			return (examples[0]['label'], len(examples), len(examples))

		# if features is empty
			# return root with most common label in examples

		if len(features) == 0:
			labels = [e['label'] for e in examples]
			most_common = Counter(labels).most_common(1)[0] # (label, count)
			return (most_common[0], most_common[1], len(examples))

		# compute gain() for all features
		# associate root with xi* (highest gain)

		gain_values = [(f, self._gain(f, examples)) for f in features]
		xi_star = max(gain_values, key=lambda x: x[1])[0]

		root['feature_name'] = xi_star['name']
		root['children'] = {}

		# for each possible value of xi*:
			# find Sv (subset of examples where xi*=v)
			# if Sv is not empty:
				# recur with Sv, and all features except xi*
				# make that subtree roots child at branch v (root.values[v] = subtree)
			# else
				# make leaf node (string, as opposed to object) of most common label in examples
				# root.values[v] = leaf, return root

		for v in xi_star['values']:
			Sv = []
			for e in examples:
				if e['values'][xi_star['name']] == v:
					Sv.append(e)
			if len(Sv) > 0:
				subtree = self._train(Sv, [f for f in features if f != xi_star])
				root['children'][v] = subtree
			else:
				labels = [e['label'] for e in examples]
				most_common = Counter(labels).most_common(1)[0]
				root['children'][v] = (most_common[0], most_common[1], len(examples))

		return root

	def _print_tree(self, root, level):
		if root == None:
			root = self.root
		for child_key in root['children'].keys():
			if type(root['children'][child_key]) is tuple: # (label, #examples with label, #examples in subset)
				leaf = root['children'][child_key]
				print ('\t' * level) + root['feature_name'] + '=' + child_key + ' ' + leaf[0] + ' ' + str(leaf[1]) + '/' + str(leaf[2])
			else:
				print ('\t' * level) + root['feature_name'] + '=' + child_key
				self._print_tree(root['children'][child_key], level + 1)


	def _classify(self, data_point, root):
		# base case
		if type(root) is tuple:
			return root[0]

		return self._classify(data_point, root['children'][data_point[root['feature_name']]])

def main():
	features = []
	training_set = []
	test_set = []
	labels = []
	diversity_func = None

	if len(sys.argv) < 5:
		print '{0}: arguments required.'.format(sys.argv[0])
		sys.exit(1)

	if sys.argv[1] == 'entropy':
		diversity_func = entropy
	elif sys.argv[1] == 'misclassification':
		diversity_func = misclassification
	elif sys.argv[1] == 'gini':
		diversity_func = gini
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
		for line in train_f.readlines():
			tokens = line.rstrip().split(',')
			example = {'label': tokens[1]}
			feature_dict = {}
			for idx, val in enumerate(tokens[2:]):
				feature_name = features[idx]['name']
				feature_dict[feature_name] = val
			example['values'] = feature_dict
			training_set.append(example)
		train_f.close()

	with open(sys.argv[4]) as test_f:
		for line in test_f.readlines():
			tokens = line.rstrip().split(',')
			data_point = {'label': tokens[1]}
			feature_dict = {}
			for idx, val in enumerate(tokens[2:]):
				feature_name = features[idx]['name']
				feature_dict[feature_name] = val
			data_point['values'] = feature_dict
			test_set.append(data_point)
		test_f.close()

	tree = ID3_Tree(labels, features, training_set, diversity_func);
	tree.train()

	print 'Using ' + sys.argv[1] + ' in Gain function:'

	training_results = [tree.classify(e['values']) == e['label'] for e in training_set]
	n_correct = Counter(training_results)[True]

	print 'The accuracy on the training set is ' + str(n_correct) + '/' + str(len(training_set)) + ' = ' + str((float(n_correct)/len(training_set)*100)) + '%'

	test_results = [tree.classify(e['values']) == e['label'] for e in test_set]
	n_correct = Counter(test_results)[True]

	print 'The accuracy on the test set is ' + str(n_correct) + '/' + str(len(test_set)) + ' = ' + str((float(n_correct)/len(test_set)*100)) + '%'

	print 'The final decision tree:'
	tree.print_tree()

if __name__ == '__main__':
	main()
