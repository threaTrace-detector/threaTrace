import time
import pandas as pd
import numpy as np
import os
import os.path as osp
import csv
def show(str):
	print (str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

os.system('tar -zxvf ../graphchi-cpp-master/graph_data/streamspot/all.tar.gz')

show('Start processing.')
data = []
gId = -1
with open('all.tsv') as f:
	tsvreader = csv.reader(f, delimiter='\t')
	for row in tsvreader:
		if int(row[5]) != gId:
			gId = int(row[5])
			show('Graph ' + str(gId))
			scene = int(gId/100)+1
			if not osp.exists('../graphchi-cpp-master/graph_data/streamspot/'+str(scene)):
				os.system('mkdir ../graphchi-cpp-master/graph_data/streamspot/'+str(scene))
			ff = open('../graphchi-cpp-master/graph_data/streamspot/'+str(scene)+'/'+str(gId)+'.txt', 'w')
		ff.write(str(row[0])+'\t'+str(row[1])+'\t'+str(row[2])+'\t'+str(row[3])+'\t'+str(row[4])+'\t'+str(row[5])+'\n')
os.system('rm all.tsv')
show('Done.')