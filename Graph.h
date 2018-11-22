/**
*   Header file to read and store Graphs
*
*   @author Ashwin Joisa
*   @author Praveen Gupta
**/
//=============================================================================================//

#include <bits/stdc++.h>
using namespace std;

class Graph {

public:

	int nodeCount, edgeCount;
	int *adjacencyList, *adjacencyListPointers;

public:

	int getNodeCount() {
		return nodeCount;
	}

	int getEdgeCount() {
		return edgeCount;
	}

	// Reads the graph and stores in Compressed Sparse Row (CSR) format
	void readGraph() {

		int u, v;
		cin >> nodeCount >> edgeCount;

		// Use vector of vectors temporarily to input graph
		vector<int> *adj = new vector<int>[nodeCount];
		for (int i = 0; i < edgeCount; i++) {
			cin >> u >> v;
			adj[u].push_back(v);
			adj[v].push_back(u);
		}

		// Copy into compressed adjacency List
		adjacencyListPointers = new int[nodeCount +1];
		adjacencyList = new int[2 * edgeCount +1];
		int pos = 0;
		for(int i=0; i<nodeCount; i++) {
			adjacencyListPointers[i] = pos;
			for(int node : adj[i])
				adjacencyList[pos++] = node;
		}
		adjacencyListPointers[nodeCount] = pos;
		delete[] adj;
	}

	int *getAdjacencyList(int node) {

		return adjacencyList;
	}

	int *getAdjacencyListPointers(int node) {

		return adjacencyListPointers;
	}
};