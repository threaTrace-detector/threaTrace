from subprocess import Popen, PIPE
import os.path as osp
import argparse
import torch
import time
import sys
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
import psutil
import os
import random

def show(*s):
	for i in range(len(s)):
		print (str(s[i]) + ' ', end = '')
	print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

fp = []
feature_num = 26*2
label_num = 8
thre = 1.0
batch_size = 2000
alert_thre = 300

exist_model = []
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

def train():
	global model
	global loader
	global data
	global device
	global fp
	global tn
	global thre
	global optimizer
	model.train()

	total_loss = 0
	for data_flow in loader(data.train_mask):
		optimizer.zero_grad()
		out = model(data.x.to(device), data_flow.to(device))
		loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * data_flow.batch_size
	return total_loss / data.train_mask.sum().item()

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
		correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()

	return correct / mask.sum().item()

def final_test(mask):
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

def train_pro(): 
	global model
	global loader
	global data
	global device
	global fp
	global tn
	global id_map_t
	global graph_id
	global optimizer
	global loop_num
	global feature_num
	global label_num
	print(data)
	loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=batch_size, shuffle=True, add_self_loops=True)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	Net = SAGENet
	model = Net(feature_num, label_num).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	test_acc = 0

	this_loop = -1
	while(1):
		this_loop += 1
		if this_loop >= loop_num: break
		model_path = '../models/'+str(graph_id)+'_'+str(this_loop)
		if not osp.exists(model_path): continue
		model.load_state_dict(torch.load(model_path))
		fp = []
		tn = []
		loss, test_acc = final_test(data.train_mask)
		print(model_path + '  loss:{:.4f}'.format(loss) + '  acc:{:.4f}'.format(test_acc) + '  fp:' + str(len(fp)))
		for i in tn:
			data.train_mask[i] = False
		if test_acc == 1: break	

	for epoch in range(60):
		
		loss = train()
		auc = test(data.train_mask)
		if epoch % 10 == 0: 
			show (epoch, loss, auc)

	max_thre = 1
	bad_cnt = 0
	while (1):
		fp = []
		tn = []
		auc = final_test(data.train_mask)
		if len(tn) == 0:
			bad_cnt += 1
		else:
			bad_cnt = 0
		if bad_cnt >= max_thre:
			break

		if len(tn) > 0:
			for i in tn:
				data.train_mask[i] = False

			fw = open('../models/fp_feature_label_'+str(graph_id)+'_'+str(loop_num)+'.txt', 'w')
			x_list = data.x[fp]
			y_list = data.y[fp]
			print(len(x_list))
			
			if len(x_list) >1:
				sorted_index = np.argsort(y_list, axis = 0)
				x_list = np.array(x_list)[sorted_index]
				y_list = np.array(y_list)[sorted_index]

			for i in range(len(y_list)):
				fw.write(str(y_list[i])+':')
				for j in x_list[i]:
					fw.write(' '+str(j))
				fw.write('\n')
			fw.close()

			fw = open('../models/tn_feature_label_'+str(graph_id)+'_'+str(loop_num)+'.txt', 'w')
			x_list = data.x[tn]
			y_list = data.y[tn]
			print(len(x_list))
			
			if len(x_list) >1:
				sorted_index = np.argsort(y_list, axis = 0)
				x_list = np.array(x_list)[sorted_index]
				y_list = np.array(y_list)[sorted_index]

			for i in range(len(y_list)):
				fw.write(str(y_list[i])+':')
				for j in x_list[i]:
					fw.write(' '+str(j))
				fw.write('\n')
			fw.close()

			torch.save(model.state_dict(),'../models/'+str(graph_id)+'_'+str(loop_num))
		
		model1 = Net(feature_num, label_num).to(device)
		model = model1
		optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
		loop_num += 1
		auc = 0
		for epoch in range(150):
			loss = train()
			auc = test(data.train_mask)
			if epoch % 10 == 0: 
				show (epoch, loss, auc)
			if auc == 1 and loss < 0.6: break
		if auc == 1: break


	fp = []
	tn = []
	auc = final_test(data.train_mask)

	for i in tn:
		data.train_mask[i] = False



	fw = open('../models/fp_feature_label_'+str(graph_id)+'_'+str(loop_num)+'.txt', 'w')
	x_list = data.x[fp]
	y_list = data.y[fp]
	print(len(x_list))
		
	if len(x_list) >1:
		sorted_index = np.argsort(y_list, axis = 0)
		x_list = np.array(x_list)[sorted_index]
		y_list = np.array(y_list)[sorted_index]

	for i in range(len(y_list)):
		fw.write(str(y_list[i])+':')
		for j in x_list[i]:
			fw.write(' '+str(j))
		fw.write('\n')
	fw.close()

	fw = open('../models/tn_feature_label_'+str(graph_id)+'_'+str(loop_num)+'.txt', 'w')
	x_list = data.x[tn]
	y_list = data.y[tn]
	print(len(x_list))

	if len(x_list) >1:
		sorted_index = np.argsort(y_list, axis = 0)
		x_list = np.array(x_list)[sorted_index]
		y_list = np.array(y_list)[sorted_index]

	for i in range(len(y_list)):
		fw.write(str(y_list[i])+':')
		for j in x_list[i]:
			fw.write(' '+str(j))
		fw.write('\n')
	fw.close()

	torch.save(model.state_dict(),'../models/'+str(graph_id)+'_'+str(loop_num))

def splitDataset():
	global trainSet
	trainSet = []

	fw = open('run_benign.sh', 'w')
	for k in range(6):
		if k == 3: continue
		now = k*100
		dataList = []
		for i in range(now, now + 100):
			dataList.append(i)
		tempTrainSet = random.sample(dataList, 75)
		tempTrainSet.sort()
		trainSet.append(tempTrainSet)
		for i in range(now, now + 100):
			if not i in tempTrainSet:
				fw.write('python -u test_streamspot.py 200000 5000 ' + str(i) + ' 1.0 >> result_benign.txt\n')
	fw.close()
	fw_a = open('run_attack.sh', 'w')
	dataList = []
	for i in range(300, 400):
		dataList.append(i)
	testSet = random.sample(dataList, 25)
	testSet.sort()
	for i in testSet:
		fw_a.write('python -u test_streamspot.py 200000 5000 ' + str(i) + ' 1.0 >> result_attack.txt\n')
	fw_a.close()

def validate(graph_id, ss):
	global loader
	global data
	global device
	global exist_model
	global fp
	global tn
	global model
	global optimizer
	scene = int(int(graph_id)/100)+1
	ans = 0
	for this_model in exist_model:
		if len(this_model) == 0: continue
		p = Popen('../graphchi-cpp-master/bin/example_apps/test file ../graphchi-cpp-master/graph_data/gdata filetype edgelist stream_file ../graphchi-cpp-master/graph_data/streamspot/' + str(scene) + '/' + str(graph_id) + '.txt batch '+ss, shell=True, stdin=PIPE, stdout=PIPE)
		ans = 0
		while (1) :
			id_map = {}
			id_map_t = {}
			ts = {}
			train_mask = []
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
					train_mask.append(True)
				else:
					train_mask.append(False)
				x.append(line[3:len(line)-1])
				ts[i] = line[len(line)-1] / 1000
				if ts[i] > this_ts: this_ts = ts[i]
			edge_num = int(p.stdout.readline())
			for i in range(edge_num):
				line = bytes.decode(p.stdout.readline())
				line =list(map(int, line.strip('\n').split(' ')))
				edge_s.append(id_map[line[0]])
				edge_e.append(id_map[line[1]])
			
			x = torch.tensor(x, dtype=torch.float)	
			y = torch.tensor(y, dtype=torch.long)
			train_mask = torch.tensor(train_mask, dtype=torch.bool)
			edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
			data = Data(x=x, y=y,edge_index=edge_index, test_mask = train_mask, train_mask = train_mask)
			dataset = TestDataset([data])
			data = dataset[0]
			loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=batch_size, shuffle=True, add_self_loops=True)
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')	
			Net = SAGENet	
			model1 = Net(feature_num, label_num).to(device)
			model = model1
			optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
			fp = []
			tn = []
			for i in this_model:
				this_loop = -1
				while(1):
					this_loop += 1
					model_path = '../models/'+str(i)+'_'+str(this_loop)
					if not osp.exists(model_path): break
					fp = []
					tn = []
					model.load_state_dict(torch.load(model_path))

					loss, test_acc = final_test(data.train_mask)
					for j in tn:
						data.train_mask[j] = False
					if len(fp) == 0: break
				if len(fp) == 0: break
			ans += len(fp)
		if ans == 0: return 0
	return ans	


def getFeature(id):
	global feature_num
	global label_num
	scene = int(int(id)/100)+1
	f = open('../graphchi-cpp-master/graph_data/streamspot/'+str(scene) + '/'+str(id)+'.txt', 'r')
	nodeType_map = {}
	edgeType_map = {}
	nodeId_map = {}
	edgeType_cnt = 0
	nodeType_cnt = 0
	for line in f:
		temp = line.strip('\n').split('\t')

		if not (temp[1] in nodeType_map.keys()):
			nodeType_map[temp[1]] = nodeType_cnt
			nodeType_cnt += 1

		if not (temp[3] in nodeType_map.keys()):
			nodeType_map[temp[3]] = nodeType_cnt
			nodeType_cnt += 1
		
		if not (temp[4] in edgeType_map.keys()):
			edgeType_map[temp[4]] = edgeType_cnt
			edgeType_cnt += 1

	f_train_feature = open('../models/feature.txt', 'w')
	for i in edgeType_map.keys():
		f_train_feature.write(str(i)+'\t'+str(edgeType_map[i])+'\n')
	f_train_feature.close()
	f_train_label = open('../models/label.txt', 'w')
	for i in nodeType_map.keys():
		f_train_label.write(str(i)+'\t'+str(nodeType_map[i])+'\n')
	f_train_label.close()
	feature_num = edgeType_cnt*2
	label_num = nodeType_cnt

def main():
	global data
	global batch_size
	global id_map
	global id_map_t
	global anomaly_node
	global ts
	global thre
	global this_ts
	global graph_id
	global loop_num
	global trainSet
	global exist_model
	anomaly_node = {}
	ss = 200000
	if len(sys.argv) > 1:
		ss = sys.argv[1]

	if len(sys.argv) > 2:
		batch_size = int(sys.argv[2])

	if len(sys.argv) > 3:
		thre = float(sys.argv[3])

	splitDataset()
	for outLoop in range(5):
		exist_model.append([])
		current_pos = len(exist_model)-1
		print('current_pos: ', current_pos)
		validated_id = []
		while(1):
			maxi = -1
			if outLoop == 0 and len(exist_model[current_pos]) == 0:
				maxi = trainSet[outLoop][0]
				getFeature(maxi)
				show('First graph: ', maxi)

			else:
				cnt = 0
				cnt_all = len(trainSet[outLoop])
				maxfp = -1
				maxi = 0
				for j in trainSet[outLoop]:
					if j in validated_id: continue
					flag = validate(j, '50000000')
					show('Graph ', j, ' validating done. fp = ', flag)
					if flag == 0: 
						validated_id.append(j)
					else:
						cnt += 1
					if flag > maxfp:
						maxfp = flag
						maxi = j
				show('Number of fp!=0 and number of all: ', cnt, cnt_all, cnt/cnt_all)
				if cnt/cnt_all < 0.05: break 

			show('Graph ' + str(maxi) + ' start training new submodels.')
			validated_id.append(maxi)
			exist_model[current_pos].append(maxi)
			loop_num = 0
			graph_id = str(maxi)
			scene = int(int(graph_id)/100)+1
			p = Popen('../graphchi-cpp-master/bin/example_apps/train file ../graphchi-cpp-master/graph_data/gdata filetype edgelist stream_file ../graphchi-cpp-master/graph_data/streamspot/' + str(scene) + '/' + graph_id + '.txt batch '+str(ss), shell=True, stdin=PIPE, stdout=PIPE)
			while (1) :
				id_map = {}
				id_map_t = {}
				ts = {}
				train_mask = []
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
						train_mask.append(True)
					else:
						train_mask.append(False)
					x.append(line[3:len(line)-1])
					ts[i] = line[len(line)-1] / 1000
					if ts[i] > this_ts: this_ts = ts[i]
				edge_num = int(p.stdout.readline())
				for i in range(edge_num):
					line = bytes.decode(p.stdout.readline())
					line =list(map(int, line.strip('\n').split(' ')))
					edge_s.append(id_map[line[0]])
					edge_e.append(id_map[line[1]])
				
				x = torch.tensor(x, dtype=torch.float)	
				y = torch.tensor(y, dtype=torch.long)
				train_mask = torch.tensor(train_mask, dtype=torch.bool)
				edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
				data = Data(x=x, y=y,edge_index=edge_index, test_mask = train_mask, train_mask = train_mask)
				dataset = TestDataset([data])
				data = dataset[0]
				train_pro()

	fw = open('models_list.txt', 'w')
	for i in exist_model:
		for j in i:
			fw.write(str(j)+' ')
		fw.write('\n')
	fw.close()
	fw = open('models_history.txt', 'a')
	for i in exist_model:
		for j in i:
			fw.write(str(j)+' ')
		fw.write('\n')
	fw.write('\n')
	fw.close()

if __name__ == "__main__":
	graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
	os.environ['GRAPHCHI_ROOT'] = graphchi_root
	main()
