tn = 0
tp = 0
fn = 0
fp = 0
eps = 1e-10
f = open('result_benign.txt', 'r')
for line in f:
	if 'fp: 0' in line:
		tn += 1
	else:
		fp += 1
f.close()
f = open('result_attack.txt', 'r')
for line in f:
	if 'fp: 0' in line:
		fn += 1
	else:
		tp += 1
f.close()
precision = tp/(tp+fp+eps)
recall = tp/(tp+fn+eps)
fscore = 2*precision*recall/(precision+recall+eps)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-Score: ', fscore)
