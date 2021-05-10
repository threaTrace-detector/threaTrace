import os.path as osp
import os
import argparse
import torch
import time
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
from data_process_test import *

def show(str):
	print (str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='0')
parser.add_argument('--scene', type=str, default='')
args = parser.parse_args()
assert args.scene in ['cadets','trace','theia','fivedirections']

thre_map = {"cadets":2.0,"trace":2.0,"theia":1.5,"fivedirections":1.5}
b_size = 5000

path = '../graphchi-cpp-master/graph_data/darpatc/' + args.scene + '_test.txt' 
os.system('cp ../groundtruth/'+args.scene+'.txt groundtruth_uuid.txt')
graphId = 1
show('Start testing graph ' + str(graphId) + ' in model '+str(args.model))
data1, feature_num, label_num, adj, adj2 = MyDataset(path, args.model)

dataset = TestDataset(data1)
data = dataset[0]

loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=b_size, shuffle=False, add_self_loops=True)


class SAGENet(torch.nn.Module):
	def __init__(self, in_channels, out_channels, concat=False):
		super(SAGENet, self).__init__()
		self.conv1 = SAGEConv(in_channels, 32, normalize=False, concat=concat)
		self.conv2 = SAGEConv(32, out_channels, normalize=False, concat=concat)

	def forward(self, x, data_flow):
		data = data_flow[0]
		x = x[data.n_id]
		x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
		x = F.dropout(x, p=0.5, training=self.training)
		data = data_flow[1]
		x = self.conv2((x, None), data.edge_index, size=data.size)
		return F.log_softmax(x, dim=1)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Net = SAGENet



model = Net(feature_num, label_num).to(device)

cnt6_1 = 0
cnt1_6 = 0
thre = thre_map[args.scene]
def test(mask):
	global cnt1_6
	global cnt6_1
	model.eval()

	correct = 0
	total_loss = 0
	for data_flow in loader(mask):

		out = model(data.x.to(device), data_flow.to(device))
		pred = out.max(1)[1]
		pro  = F.softmax(out, dim=1)
		pro1 = pro.max(1)
		for i in range(len(data_flow.n_id)):
			pro[i][pro1[1][i]] = -1
		pro2 = pro.max(1)
		for i in range(len(data_flow.n_id)):
			if pro1[0][i]/pro2[0][i] < thre:
				pred[i] = 100
		for i in range(len(data_flow.n_id)):
			if (data.y[data_flow.n_id[i]] != pred[i]):
				fp.append(int(data_flow.n_id[i]))
			else:
				tn.append(int(data_flow.n_id[i]))
		correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()

	return total_loss / mask.sum().item(), correct / mask.sum().item()


loop_num = 0
model_map = {0:0}
for j in range(1):

	test_acc = 0
	args.model = model_map[j]
	while(1):
		if loop_num > 100: break
		model_path = '../models/model_'+str(loop_num)
		if not osp.exists(model_path): 
			loop_num += 1
			continue
		model.load_state_dict(torch.load(model_path))

		fp = []
		tn = []
		loss, test_acc = test(data.test_mask)
		print(str(loop_num) + '  loss:{:.4f}'.format(loss) + '  acc:{:.4f}'.format(test_acc) + '  fp:' + str(len(fp)))
		for i in tn:
			data.test_mask[i] = False
		if test_acc == 1: break
		loop_num += 1
	if test_acc == 1: break
fw =open('alarm.txt', 'w')
fw.write(str(len(data.test_mask))+'\n')
for i in range(len(data.test_mask)):
	if data.test_mask[i] == True:
		fw.write('\n')
		fw.write(str(i)+':')
		neibor = set()
		if i in adj.keys():
			for j in adj[i]:
				neibor.add(j)
				if not j in adj.keys(): continue
				for k in adj[j]:
					neibor.add(k)
		if i in adj2.keys():
			for j in adj2[i]:
				neibor.add(j)
				if not j in adj2.keys(): continue
				for k in adj2[j]:
					neibor.add(k)
		
		for j in neibor:
			fw.write(' '+str(j))

fw.close()

show('Finish testing graph ' + str(graphId) + ' in model '+str(args.model))

