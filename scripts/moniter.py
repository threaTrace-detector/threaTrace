import psutil
import csv
import time
import sys

time.sleep(5)
f = open('pid.txt', 'r')
flag = -1
ProcessID0 = 0
ProcessID1 = 0
for line in f:
	flag += 1	
	if flag == 0: ProcessID0 = int(line.strip('\n'))
	if flag == 1: ProcessID1 = int(line.strip('\n'))

proc0 = psutil.Process(int(ProcessID0))
proc1 = psutil.Process(int(ProcessID1))
cnt = 0
mem0 = 0
cpu0 = 0
mem1 = 0
cpu1 = 0
last_mem0 = 0
last_mem1 = 0
last_cpu0 = 0
last_cpu1 = 0
while 1:
	try:
		now_mem0 = proc0.memory_percent()
		now_cpu0 = proc0.cpu_percent()
		mem0 += now_mem0
		cpu0 += now_cpu0
		now_mem1 = proc1.memory_percent()
		now_cpu1= proc1.cpu_percent()
	except:
		print('Monitoring end.')
		print('Average cpu utilization: ', (cpu0+cpu1)/(cnt*16))
		print('Average mem usage: ', (mem0+mem1)*640/cnt)
		break	
	if (now_mem1 == last_mem1) and (now_cpu1 == last_cpu1):
		print('same')
		cpu1 += 0
		mem1 += 0
	else:
		cpu1 += now_cpu1
		mem1 += now_mem1
	last_mem1 = now_mem1
	last_cpu1 = now_cpu1
	cnt += 1 
	print('cpu: ', (cpu0+cpu1)/(cnt*16))
	print('mem: ', (mem0+mem1)*640/cnt)
	time.sleep(0.1)
