#pragma once

#include <map>
#include "Headers.hpp"

extern const string format_txtg;
extern const string format_csv;

struct GraphReader {
  map<string, int> shrink_indices;
  map<int, string> inv_shrinking;
  vector<vector<int>> ReadGraph(string filename, string format);
  pair<int,vector<pair<string, string>>> ReadGraphEdges(string filename, string format);
  string GetOriginalFromMapped(int v);
  int GetMappedFromOriginal(string v);
};

pair<vector<int>, vector<int>> GetOrderAndWhInOrder(string filename, GraphReader& reader);
