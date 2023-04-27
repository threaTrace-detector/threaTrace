from subprocess import Popen, PIPE
import os.path as osp
import argparse
import torch
import time
import sys
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
import psutil
import os

def show(*s):
	for i in range(len(s)):
		print (str(s[i]) + ' ', end = '')
	print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

fp = []
model_list = []
feature_num = 0
label_num = 0
thre = 2.0
batch_size = 0
alert_thre = 300
first_alert = True

class TestDataset(InMemoryDataset):
	def __init__(self, data_list):
		super(TestDataset, self).__init__('/tmp/TestDataset')
		self.data, self.slices = self.collate(data_list)

	def _download(self):
		pass
	def _process(self):
		pass

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



def test(mask):
	global model
	global loader
	global data
	global device
	global fp
	global tn
	global thre
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

def update_benign(k):
	global anomaly_node
	global id_map_t
	global ts
	global this_ts
	global first_alert
	now_ts = this_ts
	k = id_map_t[k]
	if k in anomaly_node.keys():
		anomaly_node.pop(k)

def raise_alert(k):
	global anomaly_node
	global id_map_t
	global ts
	global this_ts
	global first_alert
	now_ts = this_ts
	k = id_map_t[k]
	if not k in anomaly_node.keys():
		anomaly_node[k] = now_ts	

def real_raise_alert():
	global anomaly_node
	global this_ts
	global first_alert
	now_ts = this_ts		
	for k in anomaly_node.keys():
		if (now_ts - anomaly_node[k]) > alert_thre:
			if first_alert == True:
				first_alert = False
	
def detect(): 
	global model
	global loader
	global data
	global device
	global fp
	global tn
	global id_map_t
	loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=batch_size, shuffle=True, add_self_loops=True)
	device = torch.device('cpu')
	Net = SAGENet

	test_acc = 0
	model = Net(feature_num, label_num).to(device)
	for j in model_list:
		loop_num = 0
		base_model = str(j)
		while(1):
			model_path = '../models/'+base_model+'_'+str(loop_num)
			if not osp.exists(model_path): break
			model.load_state_dict(torch.load(model_path))
			fp = []
			tn = []
			loss, test_acc = test(data.test_mask)
			for i in tn:
				data.test_mask[i] = False
				update_benign(i)
			if test_acc == 1: break
			loop_num += 1
		if test_acc == 1: break

	for i in fp:
		raise_alert(i)

def getFeature():
	global feature_num
	global label_num
	feature_num = 0
	label_num = 0
	f = open('../models/feature.txt', 'r')
	for line in f:
		feature_num += 1
	feature_num *= 2
	f.close()
	f = open('../models/label.txt', 'r')
	for line in f:
		label_num += 1
	f.close()	
	
def main():
	global data
	global batch_size
	global id_map
	global id_map_t
	global anomaly_node
	global ts
	global thre
	global this_ts
	global alert_thre
	global first_alert
	getFeature()
	f = open('models_list.txt', 'r')
	for line in f:
		model_list.append(line.strip('\n'))
	f.close()
	anomaly_node = {}
	if len(sys.argv) > 1: ss = sys.argv[1]
	fw = open('pid.txt', 'w')
	fw.write(str(os.getpid())+'\n')

	if len(sys.argv) > 2: batch_size = int(sys.argv[2])
	if len(sys.argv) > 3: graph_id = sys.argv[3]
	if len(sys.argv) > 4: thre = float(sys.argv[4])
	if len(sys.argv) > 5: alert_thre = float(sys.argv[5])
	p = Popen('../graphchi-cpp-master/bin/example_apps/test file ../graphchi-cpp-master/graph_data/gdata filetype edgelist stream_file ../graphchi-cpp-master/graph_data/unicornsc/' + graph_id + '.txt batch '+ss, shell=True, stdin=PIPE, stdout=PIPE)
	fw.write(str(p.pid+1)+'\n')
	fw.close()
	while (1) :
		id_map = {}
		id_map_t = {}
		ts = {}
		test_mask = []
		x = []
		y = []
		edge_s = []
		edge_e = []
		this_ts = 0
		node_num = int(p.stdout.readline())
		if node_num == -1: break
		for i in range(node_num):
			line = bytes.decode(p.stdout.readline())
			line =list(map(int, line.strip('\n').split(' ')))
			id_map[line[0]] = i
			id_map_t[i] = line[0]
			y.append(line[1])
			if line[2] == 1:
				test_mask.append(True)
			else:
				test_mask.append(False)
			x.append(line[3:len(line)-1])
			ts[i] = line[len(line)-1] / 1000
			if ts[i] > this_ts: this_ts = ts[i]
		edge_num = int(p.stdout.readline())
		for i in range(edge_num):
			line = bytes.decode(p.stdout.readline())
			line = list(map(int, line.strip('\n').split(' ')))
			edge_s.append(id_map[line[0]])
			edge_e.append(id_map[line[1]])
		
		x = torch.tensor(x, dtype=torch.float)	
		y = torch.tensor(y, dtype=torch.long)
		test_mask = torch.tensor(test_mask, dtype=torch.bool)
		edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
		data = Data(x=x, y=y,edge_index=edge_index, test_mask = test_mask, train_mask = test_mask)
		dataset = TestDataset([data])
		data = dataset[0]
		if first_alert == True:
			detect()
			real_raise_alert()

	show(str(graph_id) + ' finished. fp: ' + str(len(anomaly_node.keys())))

if __name__ == "__main__":
	graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
	os.environ['GRAPHCHI_ROOT'] = graphchi_root
	main()