# ID3 Algorithm
# CS 1675 Assignment #2
# Coleman Stavish

import sys
import math
from collections import Counter

# define gain functions

def entropy(S):
	entropy = 0
	labels = [e['label'] for e in S]
	cnt = Counter(labels)
	for label in cnt.keys():
		p_label = float(cnt[label]) / len(S)
		entropy += (-p_label * math.log(p_label, 2))
	return entropy

def misclassification(S):
	return

def gini(S):
	return

class ID3_Tree:
	def __init__(self, labels, features, examples, diversity_func):
		self.labels = labels
		self.features = features
		self.examples = examples
		self.diversity_func = diversity_func

	def _gain(self, feature, examples):
		entropy = self.diversity_func(examples)
		diversity_sum = 0
		for v in feature['values']:
			Sv = []
			for e in examples:
				if e['values'][feature['name']] == v:
					Sv.append(e)
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
			return examples[0]['label']

		# if features is empty
			# return root with most common label in examples

		if len(features) == 0:
			labels = [e['label'] for e in examples]
			return Counter(labels).most_common(1)[0][0]

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
				root['children'][v] = Counter(labels).most_common(1)[0][0]

		#print 'returning root ', root
		return root

	def train(self):
		self.root = self._train(self.examples, self.features)

	def print_tree(self, root, level):
		if root == None:
			root = self.root
		for child in root['children'].keys():
			if type(root['children'][child]) is str:
				print ('\t' * level) + root['feature_name'] + '=' + child + ' ' + str(root['children'][child])
			else:
				print ('\t' * level) + root['feature_name'] + '=' + child
				self.print_tree(root['children'][child], level + 1)

	def classify(self, object):
		return

def main():
	features = []
	examples = []
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

	tree = ID3_Tree(labels, features, examples, diversity_func);
	tree.train()
	tree.print_tree(tree.root, 0)


if __name__ == '__main__':
	main()



