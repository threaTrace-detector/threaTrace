tn = 0
tp = 0
fn = 0
fp = 0
eps = 1e-10
f = open('result_benign.txt', 'r')
for line in f:
	if int(line.split(' ')[3])>2:
		fp += 1
	else:
		tn += 1
f.close()

f = open('result_attack.txt', 'r')
for line in f:
	if int(line.split(' ')[3])>2:
		tp += 1
	else:
		fn += 1
f.close()
precision = tp/(tp+fp+eps)
recall = tp/(tp+fn+eps)
fscore = 2*precision*recall/(precision+recall+eps)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-Score: ', fscore)
