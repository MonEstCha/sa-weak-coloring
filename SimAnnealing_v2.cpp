#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <time.h>
#include <chrono>
#include <random>
#include <boost/graph/random.hpp>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/two_bit_color_map.hpp>
#include <boost/config/user.hpp>
//#include "matplotlibcpp.h"
#include "Headers.hpp"
#include "FilesOps.hpp"
#include "FlagParser.hpp"
#include "ReadTxt.hpp"

// compile: c++ -I /opt/homebrew/Cellar/boost/1.78.0_1/include SimAnnealing_v2.cpp ReadTxt.cpp FilesOps.cpp FlagParser.cpp -o SimAnnealing_v2 -Ofast -std=c++14
// including matplotlibcpp (not working):  c++ -I /opt/homebrew/Cellar/boost/1.78.0_1/include -I /Applications/Spyder.app/Contents/Resources/lib/python3.9/numpy/core/include -I /System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 SimAnnealing.cpp ReadTxt.cpp FilesOps.cpp FlagParser.cpp -o SimAnnealing -Ofast -std=c++14
// execute: ./SimAnnealing_v2 --in=yeast.csv --rad=1 --heur=wReachLeft

using namespace boost;

/*************** Globals and defintions for wReachLeft heuristic ***************/

vector<int> _wreach_szs;
vector<int> _deg;

struct Vert {
  int id;
  // overload "<" breaking ties by degrees
  bool operator<(const Vert& oth) const {
    if (_wreach_szs[id] != _wreach_szs[oth.id]) { return _wreach_szs[id] > _wreach_szs[oth.id]; }
    if (_deg[id] != _deg[oth.id]) { return _deg[id] > _deg[oth.id]; }
    return id < oth.id;
  }
};

/*************************** Some convenient functions *************************/

/**
 * Function checks if fullString ends with ending
 * @param fullString, String to check for ending
 * @param ending, ending to look for in fullString
 * @return true if check positive, false if negative
 */
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void Err() {
  cerr<<"Usage: ./SimAnnealing --in=graph.txtg --rad=radius [--o=output.txt]"<<endl;
  cerr<<"--h for help\n";
  exit(1);
}

template < typename Graph, typename VertexNamePropertyMap >
void printGraph(Graph g){
	using v_it = typename graph_traits<Graph>::vertex_iterator;
	cout << "vertices" << endl;
	pair<v_it,v_it> vs = vertices(g);
	copy(vs.st, vs.second,ostream_iterator<typename graph_traits<Graph>::vertex_descriptor>{cout, "\n"});

	using e_it = typename graph_traits<Graph>::edge_iterator;
	cout << "edges" << endl;
	pair<e_it,e_it> es = boost::edges(g);
	copy(es.first, es.second,ostream_iterator<typename graph_traits<Graph>::edge_descriptor>{cout, "\n"});
}

template < typename Graph, typename VertexNameMap >
int count_adj_vertices(typename graph_traits< Graph >::vertex_descriptor u, const Graph& g,
    VertexNameMap name_map)
{
	int ct = 0;
    typename graph_traits< Graph >::adjacency_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = adjacent_vertices(u, g); vi != vi_end; ++vi){
    	//cout << "in line 121 "<< ct << endl;
    	ct++;
    }
    return ct;
}

template < typename Graph >
void printAdjacentVerts(const Graph& g, typename graph_traits<Graph>::vertex_descriptor v){
	cout << "neighbors of " << v << ": {";
	// print
	typename graph_traits<Graph>::adjacency_iterator vit, vend;
	tie(vit, vend) = adjacent_vertices(v, g);
	copy(vit, vend, ostream_iterator<typename graph_traits<Graph>::vertex_descriptor>{cout, ", "});
	// count
	typename graph_traits<Graph>::adjacency_iterator vi, vi_end, next;
	int ct = 0;
	for (tie(vi, vi_end) = adjacent_vertices(v, g); vi != vi_end; ++vi){
		ct++;
	}
	cout << "}, ct: " << ct << endl;
}

template <typename descVec, typename VertexNameMap, typename Graph>
void printVec(const Graph& g, descVec vec, typename graph_traits< Graph >::vertex_descriptor root, VertexNameMap name_map, string desc){
	cout << desc << " of " << get(name_map, root) << ": {";
	for (typename graph_traits<Graph>::vertex_descriptor vD : vec) {
		int v = get(name_map, vD);
		cout << v << ", ";
	}
	cout << "}" << endl;
}

/*
 * Function calculates the probability of taking a new solution even if it is worse than the current one
 * @param t, the current temperature
 * @param wcolOld, the wcol of the current solution
 * @param wcolNew, the wcol of the new hypothetical solution
 * @param t_start, the maximum temperature
 * @return a value between 0 and 1
 */
float getProb_log(int t,float wcolOld, float wcolNew, int t_start){

	float wcolRatio = wcolNew - wcolOld + 1.0;
	float temp = 1.0/(float) (t_start*0.1);
	float tempRatio = (float) (log(t)) * temp;
	return exp(- wcolRatio * tempRatio);
}

/*
 * Function calculates the probability of taking a new solution even if it is worse than the current one
 * @param t, the current temperature
 * @param wcolOld, the wcol of the current solution
 * @param wcolNew, the wcol of the new hypothetical solution
 * @param t_start, the maximum temperature
 * @return a value between 0 and 1
 */
float getProb_exp(int t,float wcolOld, float wcolNew, int t_start){

	float wcolRatio = wcolNew - wcolOld + 1.0;
	float temp = 1.0/(float) (t * t);
	float tempRatio = (float) (t_start) * temp;
	return exp(- wcolRatio * tempRatio);
}

/*************************** Functions adapted from Nadara et al. (2019) **************************/

/**
 * Function calculates the vertices that contain the root vertex in their potential weakly R-reachable set for a given order
 * @param where_in_order, the position of the vertices in the order
 * @param phase_id, the current position to fill in the ordering
 * @return a vector containing all vertices that have root in their weakly R-reachable set
 */
template < typename Graph, typename VertexNameMap >
vector<typename graph_traits<Graph>::vertex_descriptor> ComputeSingleCluster(const Graph& graph,
                                 vector<int>& where_in_order,
                                 int R,
                                 vector<int>& is_forb,
                                 vector<int>& last_vis,
                                 vector<int>& dis,
                                 typename graph_traits<Graph>::vertex_descriptor u,
								 VertexNameMap name_map,
                                 int phase_id) {
	int root = get(name_map,u);
	//cout << "root: " <<root  << endl;

	vector<typename graph_traits<Graph>::vertex_descriptor> res;
	//if(root == 0) return res;
	if (!is_forb.empty() && is_forb[root]) { return {}; }
	last_vis[root] = phase_id;
	dis[root] = 0;
	vector<typename graph_traits<Graph>::vertex_descriptor> que{u};
	// breadth first search up to depth R
	for (int ii = 0; ii < (int)que.size(); ii++) {
		typename graph_traits<Graph>::vertex_descriptor cur_vD = que[ii];
		int cur_v = get(name_map, cur_vD);
		res.PB(cur_vD);
		if (dis[cur_v] == R) { continue; }
		//for (auto nei : graph[cur_v]) {
		typename graph_traits<Graph>::adjacency_iterator neiD, nei_end;
		for (tie(neiD, nei_end) = adjacent_vertices(cur_vD, graph); neiD != nei_end; ++neiD){
			int nei = get(name_map, *neiD);
			if (last_vis[nei] != phase_id && where_in_order[nei] > where_in_order[root] && (is_forb.empty() || !is_forb[nei])) {
				last_vis[nei] = phase_id;
				que.PB(*neiD);
				dis[nei] = dis[cur_v] + 1;
			}
		}
	}
	return res;
}

/**
 * Function calculates the vertices that contain the root vertex in their potential weakly R-reachable set for a given order
 * @param where_in_order, the position of the vertices in the order
 * @param phase_id, the current position to fill in the ordering
 * @return a vector containing all vertices that have root in their weakly R-reachable set
 */
template < typename Graph, typename VertexNameMap >
vector<typename graph_traits<Graph>::vertex_descriptor> ComputeSingleCluster(const Graph& graph,
                                 const vector<int>& where_in_order,
                                 int R,
                                 vector<int>& is_forb,
                                 typename graph_traits<Graph>::vertex_descriptor u,
								 VertexNameMap name_map,
                                 int phase_id) {
	int root = get(name_map,u);
	//cout << "root: " <<root  << endl;
	vector<int> last_vis(num_vertices(graph)+1);
	vector<int> dis(num_vertices(graph)+1);
	vector<typename graph_traits<Graph>::vertex_descriptor> res;
	//if(root == 0) return res;
	if (!is_forb.empty() && is_forb[root]) { return {}; }
	last_vis[root] = phase_id;
	dis[root] = 0;
	vector<typename graph_traits<Graph>::vertex_descriptor> que{u};
	// breadth first search up to depth R
	for (int ii = 0; ii < (int)que.size(); ii++) {
		typename graph_traits<Graph>::vertex_descriptor cur_vD = que[ii];
		int cur_v = get(name_map, cur_vD);
		res.PB(cur_vD);
		if (dis[cur_v] == R) { continue; }
		//for (auto nei : graph[cur_v]) {
		typename graph_traits<Graph>::adjacency_iterator neiD, nei_end;
		for (tie(neiD, nei_end) = adjacent_vertices(cur_vD, graph); neiD != nei_end; ++neiD){
			int nei = get(name_map, *neiD);
			if (last_vis[nei] != phase_id && where_in_order[nei] > where_in_order[root] && (is_forb.empty() || !is_forb[nei])) {
				last_vis[nei] = phase_id;
				que.PB(*neiD);
				dis[nei] = dis[cur_v] + 1;
			}
		}
	}
	return res;
} // Function seems to have side effects!!

/**
 * Function calculates all weakly reachable sets for the given graph and order
 * @return a vector with the weakly reachable sets
 */
template < typename Graph, typename VertexNameMap, typename descVec >
vector<descVec> ComputeAllWReach(const Graph& graph,
									 VertexNameMap name_map,
                                     vector<int>& where_in_order,
                                     int R,
                                     vector<int> is_forb, descVec dVdummy) {
	int n = num_vertices(graph);
	vector<int> last_vis(n + 1, -1);
	vector<int> dis(n + 1);
	vector<descVec> res(n + 1);
	int ct = 1;
	typename graph_traits< Graph >::vertex_iterator root, end;
	for(tie(root, end) = vertices(graph); root != end; ++root){
		descVec cluster = ComputeSingleCluster(graph, where_in_order, R, is_forb, last_vis, dis, *root, name_map, ct);

		for (typename graph_traits<Graph>::vertex_descriptor vD : cluster) {
			int v = get(name_map, vD);
			res[v].PB(*root);
		}
		ct++;
	}
	return res;
}

/*template < class IncidenceGraph, class DFSVisitor, class ColorMap>
void custom_depth_first_visit_impl(const IncidenceGraph& g,
	typename graph_traits< IncidenceGraph >::vertex_descriptor u,
	DFSVisitor& vis, // pass-by-reference here, important!
	ColorMap color, int depth)
{
	BOOST_CONCEPT_ASSERT((IncidenceGraphConcept< IncidenceGraph >));
	BOOST_CONCEPT_ASSERT((DFSVisitorConcept< DFSVisitor, IncidenceGraph >));
	typedef typename graph_traits< IncidenceGraph >::vertex_descriptor Vertex;
	BOOST_CONCEPT_ASSERT((ReadWritePropertyMapConcept< ColorMap, Vertex >));
	typedef typename property_traits< ColorMap >::value_type ColorValue;
	BOOST_CONCEPT_ASSERT((ColorValueConcept< ColorValue >));
	typedef color_traits< ColorValue > Color;
	typename graph_traits< IncidenceGraph >::out_edge_iterator ei, ei_end;

	put(color, u, Color::gray());
	vis.discover_vertex(u, g);

	cout << "vertex: " << u << endl;

	if (depth > 0)
		for (boost::tie(ei, ei_end) = out_edges(u, g); ei != ei_end; ++ei)
		{
			Vertex v = target(*ei, g);
			vis.examine_edge(*ei, g);
			ColorValue v_color = get(color, v);
			if (v_color == Color::white())
			{
				vis.tree_edge(*ei, g);
				custom_depth_first_visit_impl(g, v, vis, color, depth-1);
			}
			else if (v_color == Color::gray())
				vis.back_edge(*ei, g);
			else
				vis.forward_or_cross_edge(*ei, g);
			call_finish_edge(vis, *ei, g);
		}
	put(color, u, Color::black());
	vis.finish_vertex(u, g);
}*/


/**
 * Function calculates weakly reachable sets for the given graph and order starting from the leftMin-th pos. in the order
 * @return a vector with the weakly reachable sets
 */
template < typename Graph, typename VertexNameMap, typename descVec >
void UpdateWReach(const Graph& graph, VertexNameMap name_map, vector<descVec>& clusters_copy, vector<descVec>& clusters,
                                     vector<int>& where_in_order_copy, vector<int>& order_copy, int R,
									 vector<int>& wreach_szs_copy, descVec& orderD_copy, vector<int>& toUpd, int ol_start, int il, int t) {

	default_color_type c = white_color;

	int n = num_vertices(graph);
	typedef boost::queue<typename graph_traits<Graph>::vertex_descriptor> queue_t;
	vector<int> last_vis(n + 1);
	vector<int> dis(n + 1);
	vector<int> is_forb_dummy;

	int vName = 0;
	for(auto val: toUpd){
		if(val != 0){
			// add vi to the wreach sets of all vertices that appear in the cluster of vi
			//int vName = order_copy[i];
			int pos = where_in_order_copy[vName]-1;
			//cout << "pos: " << pos << endl;
			typename graph_traits<Graph>::vertex_descriptor vD = orderD_copy[pos];

			descVec clOld = clusters[vName];
			descVec clNew;
			if(R < 9){
				clNew = ComputeSingleCluster(graph, where_in_order_copy, R, is_forb_dummy, last_vis, dis, vD, name_map, vName);
			}else{
				vector<default_color_type> color(num_vertices(graph));
				bfs_visitor<null_visitor> vis;
				vis.initialize_vertex(vD, graph);
				queue_t Q;
				clNew = custom_breadth_first_visit(graph, vD, Q, vis, make_iterator_property_map(&color[0], get(vertex_index, graph), c), R, where_in_order_copy, name_map);
				//printVec(graph, clNew, vD, name_map, "clNew");
			}
			//cs_que_copy.erase({clOld.size(), vName});
			clusters_copy[vName] = clNew;
			//cs_que_copy.insert({clNew.size(), vName});
			// decrease size of all wreach sets of vertices in clOld
			for (typename graph_traits<Graph>::vertex_descriptor lostClV : clOld) {
				int v = get(name_map, lostClV);
				wreach_szs_copy[v]--;
			}
			// increase size of all wreach sets of vertices in clNew
			for (typename graph_traits<Graph>::vertex_descriptor gainedClV : clNew) {
				int v = get(name_map, gainedClV);
				wreach_szs_copy[v]++;
			}

			// As a result the vertices in the intersection will keep their wreach size
		}
		vName++;
	}

}

/**
 * Function calculates the sizes of the weakly reachable sets for the given graph and order
 * @return a vector with the sizes of the weakly reachable sets
 */
template < typename Graph, typename VertexNameMap, typename descVec >
vector<int> ComputeWreachSzs(const Graph& graph, vector<int>& where_in_order, int R, VertexNameMap name_map, descVec& clusters) {
	int n = num_vertices(graph);
	vector<int> wreach_sz(n+1);
	vector<int> last_vis(n + 1, -1);
	vector<int> dis(n + 1);
	vector<int> is_forb;
	int ct = 1;
	typename graph_traits< Graph >::vertex_iterator root, end;
	for(tie(root, end) = vertices(graph); root != end; ++root){
		int rootName = get(name_map, *root);
		vector<typename graph_traits<Graph>::vertex_descriptor> cluster = ComputeSingleCluster(graph, where_in_order, R, is_forb, last_vis, dis, *root, name_map, ct);
		clusters[rootName] = cluster;
		for (typename graph_traits<Graph>::vertex_descriptor v : cluster) {
		  // increase the wcol for all vertices in the cluster of root as they have
		  // root in their weakly r-reachable set
			wreach_sz[get(name_map, v)]++;
		}
		ct++;
	}
	return wreach_sz;
}

// Returns graph where u, v are connected iff dis(u, v) <= R
template < typename Graph, typename VertexNameMap, typename descVec >
Graph PowerGraph(const Graph& graph, int R, std::unordered_set<int>& forb, VertexNameMap name_map, descVec& vD_pg) {
	int n = num_vertices(graph) - 1;
	Graph pow_graph;
	typename property_map < Graph, vertex_name_t >::type name_map_pg = get(vertex_name, pow_graph);
	typename property_traits< typename property_map < Graph, vertex_name_t >::type >::value_type name;
	map<typename graph_traits<Graph>::vertex_descriptor,typename graph_traits<Graph>::vertex_descriptor> descMap;
	// for each vertex in graph add one to pow_graph and remember the mapping
	typename graph_traits< Graph >::vertex_iterator vi, vi_end;
	// add a dummy at pos. 0, since there is no shrinked id 0 and thus neither a corr. vertex descriptor
	vD_pg.push_back(-1);
	int ct = 1;
	for(tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi){

		typename graph_traits<Graph>::vertex_descriptor vd = add_vertex(pow_graph);
		descMap[*vi] = vd;
		name = ct;
		put(name_map_pg, vd, name);
		vD_pg.push_back(vd);
		ct++;
	}

	vector<int> last_vis(n + 1);
	vector<int> dis(n + 1);
	vector<int> where_in_order(n + 1);
	vector<int> is_forb;
	if (!forb.empty()) {
		is_forb.resize(n + 1);
		for (auto v : forb) {
		  is_forb[v] = 1;
		}
	}
	int ct2 = 1;
	typename graph_traits< Graph >::vertex_iterator rootD, end;
	for(tie(rootD, end) = vertices(graph); rootD != end; ++rootD){
		int root = get(name_map, *rootD);
		where_in_order[root] = -1; // hack, to make it think root is before everybody  in order

		vector<typename graph_traits<Graph>::vertex_descriptor> cluster = ComputeSingleCluster(graph, where_in_order, R, is_forb, last_vis, dis, *rootD, name_map, ct2);

		for (typename graph_traits<Graph>::vertex_descriptor v : cluster) {
			// get mapped vertices in pow_graph
			if(descMap[*rootD] != descMap[v] && !edge(descMap[v],descMap[*rootD],pow_graph).second)
				add_edge(descMap[*rootD], descMap[v], pow_graph);
		}
		where_in_order[root] = 0;
		ct2++;
	}

	return pow_graph;
}

/**
 * Function computes the degeneracy ordering for the passed graph
 * @param graph, the graph to calculate the degeneracy ordering for
 * @param R, the radius
 * @return a pair containing the degeneracy and the corresponding ordering
 */
template < typename Graph, typename VertexNameMap, typename descVec>
pair<vector<int>, vector<int>> Degeneracy(const Graph& graph, int R, VertexNameMap name_map, descVec& removed_orderD) {
	int n = num_vertices(graph);
	std::unordered_set<int> dummy_forb;
	Graph pow_graph;
	descVec vD_pg;
	// pow_graph: only vertices of distance up to R are connected; format is an adjacency matrix
	pow_graph = PowerGraph(graph, R, dummy_forb, name_map, vD_pg);
	typename property_map < Graph, vertex_name_t >::type name_map_pg = get(vertex_name, pow_graph);

	int degeneracy = 0;
	// vector with n+1=graph.size() elements
	vector<int> degree(n + 1);
	// bucket[i] contains the vertex v if it has degree i
	vector<set<int>> buckets(n + 1);

	vector<bool> already_removed(n + 1);
	// nonempty_buckets contains the degrees recurring in the graph
	set<int> nonempty_buckets;
	// init data structures
	typename graph_traits< Graph >::vertex_iterator vi, vi_end;
	for(tie(vi, vi_end) = vertices(pow_graph); vi != vi_end; ++vi){
		int deg = out_degree(*vi, pow_graph);
		int v = get(name_map_pg, *vi);
		buckets[deg].insert(v);
		nonempty_buckets.insert(deg);
		degree[v] = deg;
	}

	vector<int> removed_order;
	vector<int> where_in_order(n + 1);
	int count = 1;
	while (!nonempty_buckets.empty()) {
		// get currently lowest degree
		int wh_bucket = *(nonempty_buckets.begin());

		degeneracy = max(degeneracy, wh_bucket);
		// get current vertex, that is the first one in list of lowest degree
		int v_to_remove = *(buckets[wh_bucket].begin());

		typename graph_traits<Graph>::vertex_descriptor v_to_removeD = vD_pg[v_to_remove];
		removed_order.PB(v_to_remove);
		where_in_order[v_to_remove] = n - count + 1;
		removed_orderD.PB(v_to_removeD);
		already_removed[v_to_remove] = true;
		buckets[wh_bucket].erase(v_to_remove);
		// if all vertices with degree wh_bucket have been looked at remove the degree wh_bucket from nonempty_buckets
		if (buckets[wh_bucket].empty()) {
			nonempty_buckets.erase(wh_bucket);
		}
		// update the current vertex's neighbor
		typename graph_traits<Graph>::adjacency_iterator neiD, nei_end;
		for (tie(neiD, nei_end) = adjacent_vertices(v_to_removeD, pow_graph); neiD != nei_end; ++neiD){
			int nei = get(name_map_pg, *neiD);
			if (already_removed[nei]) {
				continue;
			}
			buckets[degree[nei]].erase(nei);
			if (buckets[degree[nei]].empty()) {
				nonempty_buckets.erase(degree[nei]);
			}
			degree[nei]--;
			assert(degree[nei] >= 0);
			// in case bucket for new degree of nei was empty
			if (buckets[degree[nei]].empty()) {
				nonempty_buckets.insert(degree[nei]);
			}
			// since degree has decreased, the vertex has to be added to the next lower bucket
			buckets[degree[nei]].insert(nei);
		}
		count++;
	}
	cout << "degeneracy: " << degeneracy << endl;
	buckets.clear();
	degree.clear();
	already_removed.clear();
	nonempty_buckets.clear();
	reverse(removed_order.begin(), removed_order.end());
	reverse(removed_orderD.begin(), removed_orderD.end());

	return {removed_order, where_in_order};
}

/**
 * Function computes the degeneracy ordering for the passed graph
 * @param graph, the graph to calculate the degeneracy ordering for
 * @param R, the radius
 * @return a pair containing the degeneracy and the corresponding ordering
 */
template < typename Graph, typename VertexNameMap, typename descVec>
pair<vector<int>, vector<int>> ByWReachLeft(const Graph& graph, int R, VertexNameMap name_map, descVec& orderD, descVec vD, vector<descVec>& clusters, vector<int>& where_in_order){

	int n = num_vertices(graph);
	//vector<int> where_in_order(n + 1, n + 1); // hacky hack to set where_in_order to n+1 for all not decided vertices
	vector<int> put(n + 1);
	vector<int> order;
	_wreach_szs.resize(n + 1, 1);
	_deg.resize(n + 1);
	vector<int> last_vis(n + 1);
	vector<int> dis(n + 1);
	// doesn't contain anything, but expected as param from ComputeSingleCluster
	vector<int> is_forb_dummy;
	// contains all vertices, changes over time
	set<Vert> verts;

	typename graph_traits< Graph >::vertex_iterator vi, vi_end;
	for(tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi){
		int j = get(name_map, *vi);
		// save degree
	    _deg[j] = out_degree(*vi, graph);
	    // push vertex to data structure
	    verts.insert({j});
	}

	chrono::duration<double> totalTime = chrono::duration<double>::zero();
	for (int i = 1; i <= n; i++) {
		// get next vertex to place in order: is first one of vertex set
		// which is ordered already by size of weakly reachable sets
	    int where_best = verts.begin()->id;
	    // erase it from set
	    verts.erase(verts.begin());
	    // remember where and that it was put in the order
	    where_in_order[where_best] = i;
	    put[where_best] = 1;
	    orderD.PB(vD[where_best]);
	    order.PB(where_best);
	    // calculate from which vertices it is weakly reachable (BFS)
		//auto clCalcStart = chrono::high_resolution_clock::now();
	    descVec cluster = ComputeSingleCluster(graph, where_in_order, R, is_forb_dummy, last_vis, dis, vD[where_best], name_map, i);
		//auto clCalcEnd = chrono::high_resolution_clock::now();
		//totalTime += clCalcEnd - clCalcStart;
	    clusters[where_best] = cluster;

	    for (typename graph_traits<Graph>::vertex_descriptor v: cluster) {
			// ignore the current vertex to be placed if found in the cluster
	    	int x = get(name_map, v);
			if (x == where_best) { continue; }
			// delete cluster vertex x in verts
			auto it = verts.find({x});
			verts.erase(it);
			// x' wreach_set now contains one more vertex
			_wreach_szs[x]++;
			// needs to be newly inserted to set
			// < was overloaded, so it is used for the insertion
			verts.insert({x});
		}
	}
	//cout << "Time elapsed total for cluster calc in ByWReachLeft: " << totalTime.count() << endl;
	return {order, where_in_order};
}


/**
 * Function identifies the weak coloring number of the passed graph and the given order
 * and all the vertices with a size of their weakly reachable sets equal to it
 */
int ComputeWcol(vector<int> wreach_szs) {
	  int res = 0;
	  for (auto x : wreach_szs) {
	    res = max(x, res);
	  }
	  return res;
}

template < typename VertexNameMap, typename descVec>
int selectSwapPartner(int rightV, const vector<int>& order, vector<int>& where_in_order, const vector<descVec>& clusters, vector<int>& selected, VertexNameMap name_map){
	for(int pos = 0; pos < where_in_order[rightV] - 1; pos++){
		int leftV = order[pos];
		if(!selected[leftV]){
			int valid = true;
			descVec cluster = clusters[leftV];
			// leftV should keep its cluster after being moved to pos. of rightV
			// thus each vertex in the cluster has to be greater than rightV with respect to the ordering
			for(auto clEl: cluster){
				int name = get(name_map, clEl);
				if(where_in_order[name] <= where_in_order[rightV]-1){
					valid = false;
					break;
				}
			}
			// leftV is suitable
			if(valid){
				selected[leftV] = 1;
				return leftV;
			}
		}
	}

	return -1;
}

/*************************** Contributed functions **************************/

/**
 * Function performs a customized Boost breadth-first-search
 */
template <class IncidenceGraph, class Buffer, class BFSVisitor, class ColorMap, typename VertexNameMap>
vector<typename graph_traits<IncidenceGraph>::vertex_descriptor> custom_breadth_first_visit
   (const IncidenceGraph& g,
    typename graph_traits<IncidenceGraph>::vertex_descriptor s,
    Buffer& Q, BFSVisitor vis, ColorMap color, int R, const vector<int>& where_in_order, VertexNameMap name_map)
{
	function_requires< IncidenceGraphConcept<IncidenceGraph> >();
	typedef graph_traits<IncidenceGraph> GTraits;
	typedef typename GTraits::vertex_descriptor Vertex;
	typedef typename GTraits::edge_descriptor Edge;
	function_requires< BFSVisitorConcept<BFSVisitor, IncidenceGraph> >();
	function_requires< ReadWritePropertyMapConcept<ColorMap, Vertex> >();
	typedef typename property_traits<ColorMap>::value_type ColorValue;
	typedef color_traits<ColorValue> Color;
	typename GTraits::out_edge_iterator ei, ei_end;

	vector<int> dis(num_vertices(g));
	vector<Vertex> cluster;

	//cout << "vertex: " << s << endl;
	put(color, s, Color::gray());
	vis.discover_vertex(s, g);
	Q.push(s);
	dis[s] = 0;
	while (! Q.empty()) {
		Vertex u = Q.top(); Q.pop();
		vis.examine_vertex(u, g);
		if (dis[u] < R){
			for (boost::tie(ei, ei_end) = out_edges(u, g); ei != ei_end; ++ei) {
				Vertex v = target(*ei, g);
				vis.examine_edge(*ei, g);
				ColorValue v_color = get(color, v);
				if (v_color == Color::white()) { // vertex is undiscovered

					vis.tree_edge(*ei, g);
					put(color, v, Color::gray());
					vis.discover_vertex(v, g);
					dis[v] = dis[u] + 1;
					//cout << "vertex: " << v << ", dis: " << dis[v] << ", parent: " << u <<  endl;
					if(where_in_order[get(name_map, v)] > where_in_order[get(name_map, s)])
						Q.push(v);
				} else {
					vis.non_tree_edge(*ei, g);
					if (v_color == Color::gray()) // vertex is discovered, but neighbors are not
						vis.gray_target(*ei, g);
					else
						vis.black_target(*ei, g); // vertex is discovered and so are neighbors
				}
			} // end for
		} // end if
		put(color, u, Color::black());
		cluster.PB(u);
		vis.finish_vertex(u, g);
	} // end while
	return cluster;
} // end custom_breadth_first_visit


int main(int argc, char** argv) {

	auto startMain = chrono::high_resolution_clock::now();

	/******************************* DEFINTIONS ********************************/

	typedef adjacency_list< listS, // Store out-edges of each vertex in a std::list
	vecS, // Store vertex set in a std::vector
	undirectedS, // The graph is directed
	property< vertex_name_t, int > // Add a vertex property
	> UndirectedGraph;
	typedef typename graph_traits<UndirectedGraph>::vertex_descriptor desc;
	typedef vector<desc> descVec;

	UndirectedGraph g; // use default constructor to create empty graph
	int n = 0;
	property_map < UndirectedGraph, vertex_name_t >::type name_map = get(vertex_name, g);
	typename property_traits< property_map < UndirectedGraph, vertex_name_t >::type >::value_type name;

	/******************************* GRAPH READ-IN ********************************/

	if (argc == 2 && string(argv[1]) == "--h") {
	  cerr<<"Usage: ./SimAnneal --in=graph.txtg --rad=radius --heur=heuristic [--sched=schedule] [--rd=rd] [--logID=id] [--o=output.txt] [--slope=slope] [--start=start] [--end=end]"<<endl;
	  cerr<<"in - input graph\n";
	  cerr<<"heur - heuristic to build start vertex ordering\n";
	  cerr<<"rad - the radius for which to determine the weak coloring number\n";
	  cerr<<"o - if you want to print order in not default output file\n";
	  cerr<<"logID - id of the log file\n";
	  cerr<<"sched - type of annealing schedule, can be either 'exp' for exponential or 'log' for logarithmic \n";
	  cerr<<"rd - lower bound for random value to which the probability of moving to a worse solution is compared\n";
	  cerr<<"slope, start, end - slope of annealing schedule, temperature range\n";
	  return 1;
	}

	GraphReader reader;
	/* Files */
	string graph_file, output_file, file_format, anneal_trace, log_file;
	/* directories */
	string graph_dir;
	/* params */
	string graph_name, rad_str = "1", heuristic = "wReachLeft", logID="log", schedule="exp";
	int ol_start;
	float start = 0.3, end = 0.7, slope = 0.002, nrSwaps = 50, rdVal_LB = 60.0;

	int R;
	try {
		FlagParser flag_parser;
		flag_parser.ParseFlags(argc, argv);
		graph_file = flag_parser.GetFlag("in", true);

		string cand_heuristic = flag_parser.GetFlag("heur", false);
		if(!cand_heuristic.empty()){
			heuristic = cand_heuristic;
		}

		string cand_sched = flag_parser.GetFlag("sched", false);
		if(!cand_sched.empty()){
			schedule = cand_sched;
		}

		string cand_rd = flag_parser.GetFlag("rd", false);
		if(!cand_rd.empty()){
			rdVal_LB = stof(cand_rd);
		}

		string cand_slope = (flag_parser.GetFlag("slope", false));
		if(!cand_slope.empty()){
			slope = stof(cand_slope);
		}

		string cand_start = flag_parser.GetFlag("start", false);
		if(!cand_start.empty()){
			start = stof(cand_start);
		}

		string cand_end = flag_parser.GetFlag("end", false);
		if(!cand_end.empty()){
			end = stof(cand_end);
		}

		string cand_nrOfSwaps = flag_parser.GetFlag("swaps", false);
		if(!cand_nrOfSwaps.empty()){
			nrSwaps = stof(cand_nrOfSwaps);
		}

		logID = flag_parser.GetFlag("logID", true);
		cout << "logID: " << logID << endl;

		// set format
		if(hasEnding(graph_file, format_txtg)){
			file_format = format_txtg;
		}
		else if(hasEnding(graph_file, format_csv)){
			file_format = format_csv;
		}

		assert(graph_file.find(file_format) == graph_file.size() - file_format.size());

		int last_slash = -1;
		for (int i = 0; i < (int)graph_file.size(); i++) {
			if (graph_file[i] == '/') {
				last_slash = i;
			}
		}
		graph_dir = graph_file.substr(0, last_slash + 1);
		graph_name = graph_file.substr(last_slash + 1, (int)graph_file.size() - file_format.size() - last_slash - 1);

		rad_str = flag_parser.GetFlag("rad", true);
		try {
			R = stoi(rad_str);
		} catch (...) {
			cerr<<"Error: Radius must be a positive integer\n";
		}

		output_file = graph_dir + "orders/" + graph_name + ".simAnneal" + rad_str + "_" + heuristic + ".txt";
		string cand_output_file = flag_parser.GetFlag("o", false);
		if (!cand_output_file.empty()) {
			output_file = cand_output_file;
		}
		flag_parser.Close();

		log_file = graph_dir + "log/" + graph_name + "_simAnneal_v2" + "_" + logID + ".txt";
	} catch (string err) {
		cerr<<"Error: "<<err<<endl;
		Err();
	}


	pair<int,vector<pair<string, string>>> res = reader.ReadGraphEdges(graph_file, file_format);
	n = res.st - 1;
	cout << "n: " << n << endl;
	descVec vD;
	// add a dummy at pos. 0, since there is no shrinked id 0 and thus neither a corr. vertex descriptor
	vD.push_back(-1);
	desc u;
	for(int v=1; v <= n; v++){
		// u = v-1, if the vertices of the input graph are denoted by integers
		u = add_vertex(g);
		name = v;
		put(name_map, u, name);
		vD.push_back(u);
		//cout << get(name_map, u) << endl;
	}

	// add edges
	for(auto e: res.nd){
		if(vD[reader.shrink_indices[e.st]] != vD[reader.shrink_indices[e.nd]])
			add_edge(vD[reader.shrink_indices[e.st]], vD[reader.shrink_indices[e.nd]], g);
	}
	cout << "edges added " << endl;

	/*pair<UndirectedGraph::edge_iterator, UndirectedGraph::edge_iterator> es = edges(g);
	copy(es.first, es.second, std::ostream_iterator<UndirectedGraph::edge_descriptor>{cout, "\n"});*/

	/******************************* SETUP OF INITIAL ORDER ********************************/

	int wcol = 0, wcol_test = 0;
	vector<int> wreach_szs(n+1,0);
	vector<int> order(n);
	descVec maxWcolVertD;
	vector<int> last_vis(n + 1);
	vector<int> dis(n + 1);
	// doesn't contain anything, but expected as param from ComputeSingleCluster
	vector<int> is_forb_dummy;
	std::unordered_set<int> forb;
	vector<int> where_in_order(n + 1, n + 1);
	descVec orderD;
	vector<descVec>  wreachSets(n + 1);
	vector<descVec> clusters(n + 1);
	descVec dVdummy(1);

	if(heuristic == "deg" || heuristic == "wReachLeft"){
		// orderD is passed by reference and modified in called functions
		if(heuristic == "deg"){
			tie(order, where_in_order) = Degeneracy(g, R, name_map, orderD);
			wreach_szs = ComputeWreachSzs(g, where_in_order, R, name_map, clusters);
		}
		else if(heuristic == "wReachLeft"){
			tie(order, where_in_order) = ByWReachLeft(g, R, name_map, orderD, vD, clusters, where_in_order);
			wreach_szs = _wreach_szs;

		}
		wcol = ComputeWcol(wreach_szs);
		wreachSets = ComputeAllWReach(g, name_map, where_in_order, R, is_forb_dummy, dVdummy);

	}
	else{
		// calculate the initial order, which simply is that of the vertex descriptor list
		// along the way we also calculate the size of the weakly R-reachable sets
		int count = 1;

		if(heuristic == "random"){
		    random_device rd;
			mt19937 mt(rd());
			iota(order.begin(), order.end(), 1);
			shuffle(order.begin(), order.end(), mt);
			for(auto el: order){
				where_in_order[el] = count;
				orderD.PB(vD[el]);
				count++;
			}

		}
		else{
			order.clear();
			graph_traits< UndirectedGraph >::vertex_iterator i, end;
			for(tie(i, end) = vertices(g); i != end; ++i){
				int iName = get(name_map, *i);
				where_in_order[iName] = count;
				orderD.PB(*i);
				order.PB(iName);
				count++;
			}
		}
		count = 1;
		graph_traits< UndirectedGraph >::vertex_iterator ii, ii_end;
		for(tie(ii, ii_end) = vertices(g); ii != ii_end; ++ii){
			descVec cluster = ComputeSingleCluster(g, where_in_order, R, is_forb_dummy, *ii, name_map, count);
			clusters[get(name_map, *ii)] = cluster;

			// for each vertex in the cluster add i to its weakly reachable set and increment the set size counter
			for(typename graph_traits<UndirectedGraph>::vertex_descriptor v: cluster){
				int memberName = get(name_map, v);
				wreach_szs[memberName]++;
				// compute wcol and remember each vertex v with wreachSets[v].size() == wcol
				if(wreach_szs[memberName] > wcol){
					wcol = wreach_szs[memberName];
				}
			}
			count++;

		}
	}

	cout << "initial order done" << endl;
	/*for(int j=0; j < order.size(); j++){
		cout << order[j] << ", ";
	}*/
	cout << endl;
	cout << "wcol at start: " << wcol << endl;
	//cout << "check: wcol = "<< ComputeWcol(ComputeWreachSzs(g, where_in_order, R, name_map)) << endl;

	int wcolStart = wcol;

	/******************************* SIMULATED ANNEALING ********************************/
	/* Try with reheating */

	/* initialize random seed: */
	srand (time(NULL));

	anneal_trace = graph_dir + "anneal_traces/" + graph_name + "_simAnneal_ssr_" + rad_str + "_trace.txt";
	ofstream trace;
	InitOfstream(trace, anneal_trace);

	fstream _log;
	_log.open(log_file, fstream::app);

	trace << "#t,il,wcol,swaps,rdVal,prob" << endl;
	trace << 1 << ",," << wcol << ",0,,0"<< endl;
	float t_start = (float) n / (float) 10;
	// start and end of outer loop
	ol_start = max(2, (int) (start*n));
	unsigned int ol_end = end*n*10;

	int trialLim = 5, t=0;
	int tLast = 0;

	int wcol_best = wcol;

	float t_max = max(1, (int) (t_start * exp(-slope*ol_start)));

	vector<pair<int, int>> temp;
	int j = 0;
	for(auto el: wreach_szs){
		temp.PB(make_pair(el, j));
		j++;
	}

	sort(temp.begin(), temp.end());

	for(int ol = ol_start; ol < ol_end; ol++){ // schedule loop
		// get current temperature according to annealing schedule
		if(schedule == "exp"){
			t = max(1, (int) (t_start * exp(-slope*ol)));
		}
		else{
			t = log(ol);
		}

		for(int il=0; il < trialLim; il++){ // trial loop
			int leftMin = n, rightMax = 0;
			// by mult. with (float) n/ (float) t, rdVal increases for lower temperatures
			float rdVal = min(0.99, (rand() % 100 + rdVal_LB) * 0.01); // (float) n/ (float) t;
			descVec orderD_copy = orderD;
			vector<descVec> clusters_copy = clusters;
			vector<int> where_in_order_copy = where_in_order, wreach_szs_copy = wreach_szs, order_copy = order;

			//auto it_left = cs_que_copy.begin();
			bool wcolInc = false;
			int left = 0, right = 0;
			vector<int> toUpd(n+1,0); vector<int> swapped(n+1,0);
			int count = 1;

			auto it_right = prev(temp.end(), 1);

			for(int swapI=1; swapI < nrSwaps; swapI++){ // swap loop

				// pick a random neighbor solution of the current solution, i.e.
				// swap a vertex with a maximum wreach set and another vertex
				// left and right denote the pos. of the vertices before the swap
				int rightV = order_copy[it_right->second];
				right = where_in_order_copy[it_right->second] - 1;

				//select left swap partner
				int left = rand() % right;

				if(left >= right){
					continue;
				}

				if(left > right){
					int temp = left;
					left = right;
					right = temp;

				}

				leftMin = min(leftMin, left);
				rightMax = max(rightMax, right);
				// where_in_order has to be adapted alongside swapping
				swap(orderD_copy[left], orderD_copy[right]);
				swap(order_copy[left], order_copy[right]);
				// update where_in_order_copy and collect vertices that are in cluster of swapped vertices
				for(auto el: {left, right}){
					where_in_order_copy[order_copy[el]] = el+1;
					int v = order_copy[el];
					swapped[v] = 1;
					descVec cluster = clusters[v];
					for(auto clEl: cluster){
						int name = get(name_map, clEl);
						toUpd[name] = 1;
					}
				}

				// move to a later/earlier position
				//it_left = next(it_left, 1);
				it_right = prev(it_right, 1);
				count++;

			} // end swap loop

			// collect vertices that are weakly reachable from swapped vertices
			int v = 0;
			for(auto val: toUpd){
				int wio = where_in_order_copy[v];
				if(val == 1){
					if(wio - 1 < leftMin || wio - 1 > rightMax){
						toUpd[v] = 0;
					}
				}
				else if(v!=0){
					descVec _cluster = clusters[v];
					for(auto clEl: _cluster){
						int name = get(name_map, clEl);
						if(swapped[name] == 1 && wio - 1 >= leftMin){
							toUpd[v] = 1;
							break;
						}
					}
				}
				v++;

			}

			// do all updates

			UpdateWReach(g, name_map, clusters_copy, clusters, where_in_order_copy, order_copy,
					R, wreach_szs_copy, orderD_copy, toUpd, ol_start, il, t);

			int wcolNew = ComputeWcol(wreach_szs_copy);

			wcolInc = wcolNew > wcol;
			if(wcolNew < wcol_best){
				wcol_best = wcolNew;
			}

			// the difference between the old and new wcol increases approx. quadratically
			// make use of distributivity: each value increases approx. quadratically
			int wcolNorm = wcol/(R*R);
			int wcolNewNorm = wcolNew/(R*R);
			float prob = 0.0;
			if(wcolInc){
				if(schedule == "exp"){
					prob = min(0.99, getProb_exp(t, wcolNorm, wcolNewNorm, t_max)*1.0);
				}
				else{
					prob = min(0.99, getProb_log(t,wcolNorm, wcolNewNorm, n)*1.0);
				}
			}

			if(!wcolInc || prob >= rdVal){
				// found better solution or accepting worse one
				// update wcol and data structures
				wcol = wcolNew;
				where_in_order = where_in_order_copy, wreach_szs = wreach_szs_copy, clusters = clusters_copy;
				orderD = orderD_copy, order = order_copy;

				temp.clear();
				int j = 0;
				for(auto el: wreach_szs){
					temp.PB(make_pair(el, j));
					j++;
				}

				sort(temp.begin(), temp.end());

				if(wcolInc && prob >= rdVal){
					trace << t << "," << il << "," << wcol << "," << nrSwaps << "," << rdVal << "," << prob << endl;
				}
				else{
					trace << t << "," << il << "," << wcol << "," << nrSwaps << endl;
				}
				break; // trial loop
			}

		} // end trial loop

		//tLast = t;
		if(ol % int (end * n) == 0){
			cout << "iteration: " << ol << ", wcol: " << wcol << endl;
		}
	} // end schedule loop

	trace.close();

	for(int j=0; j < order.size(); j++){
		assert(order[j] = orderD[j] + 1);
	}
	ofstream out;
	InitOfstream(out, output_file);
	cout << "Order final!" <<endl;

	cout << "wcol best: "<< wcol_best << endl;
	// check consistency
	vector<int> ws = ComputeWreachSzs(g, where_in_order, R, name_map, clusters);
	cout << "check: wcol = "<< ComputeWcol(ws) << endl;
	for(int v=1; v<=n; v++){
		//cout << wreach_szs[v] << "/" << ws[v] << ", ";
		assert(wreach_szs[v] == ws[v]);
	}
	for (int v : order) {
		//cout<< v <<endl;
		out << reader.GetOriginalFromMapped(v) << " ";
	}
	out << endl;

	out.close();

	auto endMain = chrono::high_resolution_clock::now();
	chrono::duration<double> totalTimeMain = endMain - startMain;

	_log << "   " << slope << ",   " << start << ",   " << end << ",   " << wcolStart << ",   " << wcol_best << ",   " << totalTimeMain.count() << ",   " << trialLim << endl;
	_log.close();


	cout << "total time elapsed: " << totalTimeMain.count() << endl;
}
