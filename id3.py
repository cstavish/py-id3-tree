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


if __name__ == '__main__':

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
	


