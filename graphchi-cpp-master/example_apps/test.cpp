#include <string>
#include <map>
#include <set>
#include "graphchi_basic_includes.hpp"
#include "engine/dynamic_graphs/graphchi_dynamicgraph_engine.hpp"
#include "util/toplist.hpp"
#include "api/dynamicdata/chivector.hpp"
#include  <unistd.h>
#include  <stdio.h>
#include <pthread.h> 
#define DYNAMICEDATA 1
using namespace graphchi;
using std::cout;
using std::vector;
using std::set;
using std::map;
using std::cin;
using std::numeric_limits;
using std::streamsize;
using std::endl;
using std::string;
pthread_mutex_t my_mutex;
pthread_mutex_t my_mutex2;

pthread_barrier_t graph_barrier;
pthread_barrier_t start_barrier;

char current_path[200]; 

/**
* Type definitions. Remember to create suitable graph shards using the
* Sharder-program.
*/
#define max_feature_num 100

int feature_num;

struct node_data {
	int label = -1;
	int id = -1;
	int under_detect = -1;
	int add_flag = -1;
	int ts = -1;
	unsigned long feature[max_feature_num*2] ={ 0 };
	bool operator<(const node_data &b) const
	{
		return id < b.id;
	}

};

struct edge_data {
	int srcType = -1;
	int dstType = -1;
	int edgeType= -1;
	int is_new = -1;
	int ts = -1;
};

typedef node_data VertexDataType;
typedef edge_data EdgeDataType;

/**
* GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
* class. The main logic is usually in the update function.
*/

std::string stream_file;

bool final_run;
bool first_run;
bool end_signal;
int final_cnt = 0;
set <node_data> node_transfer;
vector <int> edge_transfer_src;
vector <int> edge_transfer_dst;
graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType>* engine;
struct MyGraphChiProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {


	/**
	*  Vertex update function.
	*/
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
		if (final_run) {
			if (gcontext.iteration  == 210) final_cnt+=1;
			//end_signal = true;
			//ans[vertex.id()] = vertex.get_data();
			return;
		}

		//cout << gcontext.iteration << ": " << vertex.id() << endl;
		if (gcontext.iteration == 0) {
			gcontext.scheduler->add_task(0);
		}
		else {
			int this_iteration = (gcontext.iteration - 2) % 7;
			if (this_iteration == -1) this_iteration += 7;
			if (this_iteration % 7 == 0) {
				node_data now_node = vertex.get_data();
				//cout << vertex.id() << endl;
				now_node.id = vertex.id();
				for (int i=0; i<vertex.num_outedges(); i++) {
					edge_data now_edge = vertex.outedge(i)->get_data();
					if (now_edge.is_new > 0) {
						now_node.ts = now_edge.ts;
						now_edge.is_new -= 1;
						now_node.label = now_edge.srcType;
						now_node.feature[now_edge.edgeType] += 1;
						vertex.outedge(i)->set_data(now_edge);
					}
				}
				vertex.set_data(now_node);
				gcontext.scheduler->add_task(vertex.id());

			}
			else if (this_iteration % 7 == 1) {
				node_data now_node = vertex.get_data();
				for (int i=0; i<vertex.num_inedges(); i++) {
					edge_data now_edge = vertex.inedge(i)->get_data();
					if (now_edge.is_new > 0) {
						now_node.ts = now_edge.ts;
						now_edge.is_new -= 1;
						now_node.label = now_edge.dstType;
						now_node.feature[feature_num + now_edge.edgeType] += 1;
						vertex.inedge(i)->set_data(now_edge);
					}
				}
				vertex.set_data(now_node);
				gcontext.scheduler->add_task(now_node.id);
			}
			else if (this_iteration % 7 == 2) {
				gcontext.scheduler->add_task(vertex.id());
				for (int i=0; i<vertex.num_outedges(); i++) {
					gcontext.scheduler->add_task(vertex.outedge(i)->vertex_id());
				}
			}
			else if (this_iteration % 7 == 3) {
				gcontext.scheduler->add_task(vertex.id());
				for (int i=0; i<vertex.num_outedges(); i++)
					gcontext.scheduler->add_task(vertex.outedge(i)->vertex_id());
			}
			else if (this_iteration % 7 == 4) {
				node_data now_node = vertex.get_data();
				now_node.add_flag = gcontext.iteration;
				vertex.set_data(now_node);
				now_node.under_detect = 1;

				pthread_mutex_lock(&my_mutex);

				node_transfer.insert(now_node);
				pthread_mutex_unlock(&my_mutex);
				for (int i=0; i<vertex.num_inedges(); i++) {
					gcontext.scheduler->add_task(vertex.inedge(i)->vertex_id());
					pthread_mutex_lock(&my_mutex2);
					if (vertex.inedge(i)->vertex_id()>1 && vertex.id() > 1) {
						edge_transfer_src.push_back(vertex.inedge(i)->vertex_id());
						edge_transfer_dst.push_back(vertex.id());
					}
					pthread_mutex_unlock(&my_mutex2);
					
				}
			}
			else if (this_iteration % 7 == 5) {
				node_data now_node = vertex.get_data();
				pthread_mutex_lock(&my_mutex);
				node_transfer.insert(now_node);
				pthread_mutex_unlock(&my_mutex);
				for (int i=0; i<vertex.num_inedges(); i++) {
					gcontext.scheduler->add_task(vertex.inedge(i)->vertex_id());
					if (now_node.add_flag == gcontext.iteration - 1) continue;
					pthread_mutex_lock(&my_mutex2);
					if (vertex.inedge(i)->vertex_id()>1 && vertex.id() > 1) {
						edge_transfer_src.push_back(vertex.inedge(i)->vertex_id());
						edge_transfer_dst.push_back(vertex.id());
					}
					pthread_mutex_unlock(&my_mutex2);
				}
				gcontext.scheduler->add_task(0);
			}
			else if (this_iteration % 7 == 6) {
				node_data now_node = vertex.get_data();
				pthread_mutex_lock(&my_mutex);
				node_transfer.insert(now_node);
				pthread_mutex_unlock(&my_mutex);
				gcontext.scheduler->add_task(0);
			}
		}
	}

	/**
	* Called before an iteration starts.
	*/
	void before_iteration(int iteration, graphchi_context &gcontext) {
	}

	/**
	* Called after an iteration has finished.
	*/
	void after_iteration(int iteration, graphchi_context &gcontext) {
		logstream(LOG_DEBUG) << "Current Iteration: " << iteration << std::endl;
		//if (iteration ==211) cout << final_cnt << endl;
		if (iteration != 0) {
			int this_iteration = (gcontext.iteration - 2) % 7;
			if (this_iteration == -1) this_iteration += 7;
			if (this_iteration % 7 == 6) {

				if (gcontext.iteration == 1) {
					node_transfer.clear();
					edge_transfer_src.clear();
					edge_transfer_dst.clear();
				}
				else {
					char buf[1024];
					sprintf(buf, "%d", iteration );
					std::string temp_buf = buf;	
					//cout << edge_transfer_src.size() << endl;
					std::string fw_path = "feature_graphchi_unicorn_" + temp_buf;
					//freopen(fw_path.c_str(), "w", stdout);
					
					set<node_data>::iterator iter = node_transfer.begin();
					cout << node_transfer.size()-1 << endl;
					while (iter!=node_transfer.end())
					{
						if (iter->id == 0) {
							iter++;
							continue;
						}
						cout << iter->id << ' ' << iter->label << ' ' << iter->under_detect;
						for (int i=0; i<feature_num*2; i++)
							cout << ' ' << iter->feature[i];
						cout << ' ' << iter->ts << endl;
						//cout << endl;
						iter++;
					}
					fw_path = "edge_graphchi_unicorn_" + temp_buf;
					//freopen(fw_path.c_str(), "w", stdout);
					cout << edge_transfer_src.size() << endl;
					for (int i=0; i<edge_transfer_src.size(); i++) {
						cout << edge_transfer_src[i] << ' ' << edge_transfer_dst[i] <<endl;
					}
					node_transfer.clear();
					edge_transfer_src.clear();
					edge_transfer_dst.clear();
				}
				pthread_barrier_wait(&start_barrier);
				//if (end_signal == false)
				pthread_barrier_wait(&graph_barrier);
			}
		}
	}
	/**
	* Called before an execution interval is started.
	*/
	void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {
	}

	/**
	* Called after an execution interval has finished.
	*/
	void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &gcontext) {
	}

};

int node_cnt;
map<std::string, int> nodeTypeMap;
map<std::string, int> edgeTypeMap;
map<std::string, int> nodeIdMap;

void preProcess() {
	std::string label_file = "../models/label.txt";
	std::string feature_file = "../models/feature.txt";

	FILE * f1 = fopen(label_file.c_str(), "r");
	char s[1024];
	char delims[] = ":\t ";
	std::string t;
	while (fgets(s, 1024, f1) != NULL) {
		int t1;
		t  = strtok(s, delims);
		t1  = atoi(strtok(NULL, delims));
		nodeTypeMap[t] = t1;
	}
	fclose(f1);
	/*
	map<std::string,int>::iterator strmap_iter = nodeTypeMap.begin();
	for(;strmap_iter !=nodeTypeMap.end();strmap_iter++)
	{
	cout<<strmap_iter->first<<' '<<strmap_iter->second<<endl;
	}
	cout<<endl;
	*/
	FILE * f2 = fopen(feature_file.c_str(), "r");
	feature_num = 0;
	while (fgets(s, 1024, f2) != NULL) {
		feature_num += 1;
		int t1;
		t  = strtok(s, delims);
		t1  = atoi(strtok(NULL, delims));
		edgeTypeMap[t] = t1;
	}
	fclose(f2);
	return;
}

int batch_num;

void * dynamic_graph_reader(void * info) {

	FILE * f = fopen(stream_file.c_str(), "r");
	if (f == NULL) {
		logstream(LOG_ERROR) << "Unable to open the file for streaming: " << stream_file << ". Error code: " << strerror(errno) << std::endl;
	}
	//assert(f != NULL);

	int srcId, dstId, srcType, dstType, edgeType;
	//vid_t tss;
	int cnt = 0;
	char s[1024];
	edge_data temp;
	char delims[] = ":\t ";
	node_cnt = 2;


	while (fgets(s, 1024, f) != NULL) {
		if (cnt % batch_num == 0) pthread_barrier_wait(&start_barrier);
		FIXLINE(s);
		//cin >> srcId >> srcType >> dstId >> dstType >> edgeType;
		std::string t1, t2, t3, t4, t5, t6;
		t1  = strtok(s, delims);

		t2 = strtok(NULL, delims);
		t3  = strtok(NULL, delims);
		t4 = strtok(NULL, delims);
		t5 = strtok(NULL, delims);
		t6 = strtok(NULL, delims);

		if (nodeTypeMap.find(t2) == nodeTypeMap.end())
			continue;
		if (nodeTypeMap.find(t4) == nodeTypeMap.end())
			continue;
		if (edgeTypeMap.find(t5) == edgeTypeMap.end())
			continue;
		
		srcType = nodeTypeMap[t2];
		dstType = nodeTypeMap[t4];
		edgeType = edgeTypeMap[t5];

		if (nodeIdMap.find(t1) == nodeIdMap.end()) {
			nodeIdMap[t1] = node_cnt;
			node_cnt += 1;
		}
		srcId = nodeIdMap[t1];

		if (nodeIdMap.find(t3) == nodeIdMap.end()) {
			nodeIdMap[t3] = node_cnt;
			node_cnt += 1;
		}
		dstId = nodeIdMap[t3];
		int  tss = atoi(t6.c_str());

		temp.srcType = srcType;
		temp.dstType = dstType;
		temp.edgeType = edgeType;
		temp.ts = tss;
		temp.is_new = 2;
		cnt += 1;
		bool success = false;
		while (!success) {
			success = engine->add_edge(srcId, dstId, temp);
		}
		engine->add_task(srcId);
		engine->add_task(dstId);
		if (cnt % batch_num == 0) {
			//cout << cnt << endl;
			pthread_barrier_wait(&graph_barrier);
		}

	}
	fclose(f);
	if (cnt % batch_num == 0) {
		pthread_barrier_wait(&start_barrier);
		engine->add_task(0);
	}
	//usleep(1000000);
	//cout << cnt << endl;
	pthread_barrier_wait(&graph_barrier);
	pthread_barrier_wait(&start_barrier);
	final_run = true;
	for (int i=2; i<engine->num_vertices(); i++)
		engine->add_task(i);
	pthread_barrier_wait(&graph_barrier);
	//engine->finish_after_iters(2);
	return NULL;
}

int main(int argc, const char ** argv) {

	getcwd(current_path, 200);
	pthread_mutex_init(&my_mutex, NULL);
	pthread_mutex_init(&my_mutex2, NULL);
	preProcess();

	//return 0;


	graphchi_init(argc, argv);	

	metrics m("my-application-name");
	global_logger().set_log_level(5);

	std::string filename = get_option_string("file");  // Base filename
	int niters           = 1000000; // Number of iterations
	stream_file = get_option_string("stream_file");
	batch_num = get_option_int("batch");
	bool scheduler          = true; // Whether to use selective scheduling

	/* Detect the number of shards or preprocess an input to create them */
	int nshards          = convert_if_notexists<EdgeDataType>(filename,
		get_option_string("nshards", "auto"));

	/* Run */
	MyGraphChiProgram program;
	engine = new graphchi_dynamicgraph_engine<VertexDataType, EdgeDataType>(filename, nshards, scheduler, m);
	//engine->set_modifies_inedges(false); // Improves I/O performance.

	pthread_barrier_init(&start_barrier, NULL, 2);
	pthread_barrier_init(&graph_barrier, NULL, 2);
	pthread_t strthread;
	int ret = pthread_create(&strthread, NULL, dynamic_graph_reader, NULL);
	final_run = false;
	end_signal = false;

	engine->run(program, niters);

	int ret_stream = pthread_barrier_destroy(&start_barrier);
	int ret_graph = pthread_barrier_destroy(&graph_barrier);
	if (ret_stream == EBUSY) {
		logstream(LOG_ERROR) << "stream_barrier cannot be destroyed." << std::endl;
	}
	if (ret_graph == EBUSY) {
		logstream(LOG_ERROR) << "graph_barrier cannot be destroyed." << std::endl;
	}
	pthread_mutex_destroy(&my_mutex);
	pthread_mutex_destroy(&my_mutex2);
	cout << -1 << endl;
	/* Report execution metrics */
	//metrics_report(m);
	return 0;
}