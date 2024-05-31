#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <random>
#include "ReadTxt.hpp"
#include "FilesOps.hpp"

/*#include </opt/homebrew/Cellar/boost/1.78.0_1/include/boost/algorithm/string.hpp>
#include </opt/homebrew/Cellar/boost/1.78.0_1/include/boost/algorithm/string/std_containers_traits.hpp>*/
const string format_txtg = ".txtg";
const string format_csv = ".csv";
const string format_dat = ".dat";

pair<int,vector<pair<string, string>>> GraphReader::ReadGraphEdges(string filename, string format) {
  ifstream in;
  InitIfstream(in, filename);
  string line;
  vector<pair<string, string>> edges;
  string a, b;
  int count = 0;

  if(format.compare(format_csv) == 0){
	 while (getline(in, line)) {
	  // Assuming that csv files have the columns: Source,Target,Type,Id,Label,Weight
		 if (line[0] == 'S') {
			  continue;
		  }
		  vector<string> result;
		  std::stringstream ss(line);
		  while(ss.good()){
			  string substr;
			  getline( ss, substr, ',' );
			  result.push_back(substr);
		  }
		  a = result[0], b = result[1];

		  shrink_indices[a] = shrink_indices[b] = 1;
		  edges.push_back({a, b});
		  count++;
	 }
  }
  else{
	 while (getline(in, line)) {
		if (line[0] == '#') { continue; }
		stringstream stream(line);
		stream >> a >> b;
		shrink_indices[a] = shrink_indices[b] = 1;
		edges.push_back({a, b});
	 }

  }

  in.close();
  int n = 0;
  for (auto& p : shrink_indices) {
    n++;
    p.nd = n;
    inv_shrinking[p.nd] = p.st;
  }
  //isolated vertices not counted as they don't matter

  return {n+1, edges};
}


vector<vector<int>> GraphReader::ReadGraph(string filename, string format) {
  ifstream in;
  InitIfstream(in, filename);
  string line;
  vector<pair<string, string>> edges;
  string a, b;
  int count = 0;

  if(format.compare(format_csv) == 0){
	  //debug(format);
	 while (getline(in, line)) {
	  // Assuming that csv files have the columns: Source,Target,Type,Id,Label,Weight
		 if (line[0] == 'S') {
			  continue;
		  }
		  vector<string> result;
		  std::stringstream ss(line);
		  while(ss.good()){
			  string substr;
			  getline( ss, substr, ',' );
			  result.push_back(substr);
		  }
		  a = result[0], b = result[1];
		  // inserts ordered by string
		  shrink_indices[a] = shrink_indices[b] = 1;
		  edges.push_back({a, b});
		  count++;
	 }
  }
  else{
	 while (getline(in, line)) {
		if (line[0] == '#') { continue; }
		stringstream stream(line);
		stream >> a >> b;
		//if(count < 10)
		//	cout << "line " << count << ": " << line << endl;
		shrink_indices[a] = shrink_indices[b] = 1;
		edges.push_back({a, b});
		count++;
	 }

  }

  in.close();
  int l = 0;
  random_device rd;
  mt19937 mt(rd());
  int n = shrink_indices.size();
  vector<int> random_list(n);
  iota(random_list.begin(), random_list.end(), 1);
  shuffle(random_list.begin(), random_list.end(), mt);

  for (auto& p : shrink_indices) {
	// maps ordered ascendingly by original id
    // p.st is original and p.nd mapped value
    p.nd = random_list[l];
    //p.nd = n;
    inv_shrinking[p.nd] = p.st;
    l++;
  }
  vector<vector<int>> graph(n + 1);
  for (auto e : edges) {
    graph[shrink_indices[e.nd]].PB(shrink_indices[e.st]);
    graph[shrink_indices[e.st]].PB(shrink_indices[e.nd]);
  }
  return graph;
}

/**
 * Inverts friom "shrinked" id to original one
 */
string GraphReader::GetOriginalFromMapped(int ind) {
  if (inv_shrinking.count(ind) == 0) { assert(false); }
  return inv_shrinking[ind];
}

/**
 * Maps vertex id to smaller one, i.e. users numbers from 1 to n if there are n vertices
 */
int GraphReader::GetMappedFromOriginal(string ind) {
  if (shrink_indices.count(ind) == 0) { assert(false); }
  return shrink_indices[ind];
}

pair<vector<int>, vector<int>> GetOrderAndWhInOrder(string filename, GraphReader& reader) {
  int n = reader.shrink_indices.size();
  vector<int> order;
  ifstream oin;
  InitIfstream(oin, filename);
  vector<int> where_in_order(n + 1);
  string v;
  int i = 0;
  while (oin >> v) {
    int mapped = reader.GetMappedFromOriginal(v);
    assert(mapped != -1 && where_in_order[mapped] == 0);
    order.PB(mapped);
    where_in_order[mapped] = i;
    i++;
  }
  oin.close();
  return {order, where_in_order};
}
