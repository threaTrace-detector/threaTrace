import os.path as osp
import argparse
import torch
import time
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset

class TestDataset(InMemoryDataset):
	def __init__(self, data_list):
		super(TestDataset, self).__init__('/tmp/TestDataset')
		self.data, self.slices = self.collate(data_list)

	def _download(self):
		pass
	def _process(self):
		pass

def MyDataset(path, model):
	graphId = model
	node_cnt = 0
	nodeType_cnt = 0
	edgeType_cnt = 0
	provenance = []
	nodeType_map = {}
	edgeType_map = {}
	edge_s = []
	edge_e = []
	data_thre = 1000000

	for out_loop in range(1):
		f = open(path, 'r')

		nodeId_map = {}

		for line in f:
			temp = line.strip('\n').split('\t')
			if not (temp[0] in nodeId_map.keys()):
				nodeId_map[temp[0]] = node_cnt
				node_cnt += 1
			temp[0] = nodeId_map[temp[0]]	

			if not (temp[2] in nodeId_map.keys()):
				nodeId_map[temp[2]] = node_cnt
				node_cnt += 1
			temp[2] = nodeId_map[temp[2]]

			if not (temp[1] in nodeType_map.keys()):
				nodeType_map[temp[1]] = nodeType_cnt
				nodeType_cnt += 1
			temp[1] = nodeType_map[temp[1]]

			if not (temp[3] in nodeType_map.keys()):
				nodeType_map[temp[3]] = nodeType_cnt
				nodeType_cnt += 1
			temp[3] = nodeType_map[temp[3]]
			
			if not (temp[4] in edgeType_map.keys()):
				edgeType_map[temp[4]] = edgeType_cnt
				edgeType_cnt += 1

			temp[4] = edgeType_map[temp[4]]
			edge_s.append(temp[0])
			edge_e.append(temp[2])
			provenance.append(temp)

	f_train_feature = open('../models/feature.txt', 'w')
	for i in edgeType_map.keys():
		f_train_feature.write(str(i)+'\t'+str(edgeType_map[i])+'\n')
	f_train_feature.close()
	f_train_label = open('../models/label.txt', 'w')
	for i in nodeType_map.keys():
		f_train_label.write(str(i)+'\t'+str(nodeType_map[i])+'\n')
	f_train_label.close()
	feature_num = edgeType_cnt
	label_num = nodeType_cnt

	x_list = []
	y_list = []
	train_mask = []
	test_mask = []
	for i in range(node_cnt):
		temp_list = []
		for j in range(feature_num*2):
			temp_list.append(0)
		x_list.append(temp_list)
		y_list.append(0)
		train_mask.append(True)
		test_mask.append(True)
	for temp in provenance:
		srcId = temp[0]
		srcType = temp[1]
		dstId = temp[2]
		dstType = temp[3]
		edge = temp[4]
		x_list[srcId][edge] += 1
		y_list[srcId] = srcType
		x_list[dstId][edge+feature_num] += 1
		y_list[dstId] = dstType

	x = torch.tensor(x_list, dtype=torch.float)	
	y = torch.tensor(y_list, dtype=torch.long)
	train_mask = torch.tensor(train_mask, dtype=torch.bool)
	test_mask = train_mask
	edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
	data1 = Data(x=x, y=y,edge_index=edge_index, train_mask=train_mask, test_mask = test_mask)
	feature_num *= 2
	return [data1], feature_num, label_num,0,0