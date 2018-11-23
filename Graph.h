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
	int *edgeList1, *edgeList2;

public:

	int getNodeCount() {
		return nodeCount;
	}

	int getEdgeCount() {
		return edgeCount;
	}

	// Reads the graph and stores in Compressed Sparse Row (CSR) format
	void readGraph() {

		// Reading in CSR format
		cin >> nodeCount >> edgeCount;

		// Copy into compressed adjacency List
		adjacencyListPointers = new int[nodeCount +1];
		adjacencyList = new int[2 * edgeCount +1];

		for(int i=0; i<=nodeCount; i++) 
			cin >> adjacencyListPointers[i];

		for(int i=0; i<(2 * edgeCount); i++)
			cin >> adjacencyList[i];
	}

	void convertToCOO() {
		edgeList2 = adjacencyList;
		edgeList1 = new int[2 * edgeCount +1];

		for(int i=0; i <nodeCount; ++i) {
			for(int j=adjacencyListPointers[i]; j<adjacencyListPointers[i+1]; ++j){
				edgeList1[j] = i;
			}
		}
	}

	int *getAdjacencyList(int node) {

		return adjacencyList;
	}

	int *getAdjacencyListPointers(int node) {

		return adjacencyListPointers;
	}
};