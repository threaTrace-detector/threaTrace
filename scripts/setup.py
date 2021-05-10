import os

#step1: define environment variable GRAPHCHI_ROOT
#############################################
graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
os.environ['GRAPHCHI_ROOT'] = graphchi_root



#step2: clean models directory ../models
#############################################
model_dir = '../models'
models = os.listdir(model_dir)
for i in models:
	path = os.path.join(model_dir,i)
	os.system('rm ' + path)
os.system('rm result_*')