/**
*   Serial implementation for Accelerating Graph Betweenness Centrality for Sparse Graphs
*
*   @author Ashwin Joisa
*   @author Praveen Gupta
**/
//=============================================================================================//

#include "Graph.h"
using namespace std;

//=============================================================================================//

double *betweennessCentrality(Graph *graph)
{
    const int nodeCount = graph->getNodeCount();

    double *bwCentrality = new double[nodeCount]();

    for (int s = 0; s < nodeCount; s++)
    {
        stack<int> st;
        vector<int> *predecessor = new vector<int>[nodeCount];
        int *sigma = new int[nodeCount]();
        int *distance = new int[nodeCount];

        //FIXME: Change to initialize without O(V)
        memset(distance, -1, nodeCount * sizeof(int));

        distance[s] = 0;
        sigma[s] = 1;
        queue<int> q;
        q.push(s);
        while (!q.empty())
        {
            int v = q.front();
            q.pop();
            st.push(v);

            // For each neighbour w of v
            for (int i = graph->adjacencyListPointers[v]; i < graph->adjacencyListPointers[v + 1]; i++)
            {
                int w = graph->adjacencyList[i];
                // If w is visited for the first time
                if (distance[w] < 0)
                {
                    q.push(w);
                    distance[w] = distance[v] + 1;
                }
                // If shortest path to w from s goes through v
                if (distance[w] == distance[v] + 1)
                {
                    sigma[w] += sigma[v];
                    predecessor[w].push_back(v);
                }
            }
        }

        double *dependency = new double[nodeCount]();

        // st returns vertices in order of non-increasing distance from s
        while (!st.empty())
        {
            int w = st.top();
            st.pop();

            for (const int &v : predecessor[w])
            {
                if (sigma[w] != 0)
                    dependency[v] += (sigma[v] * 1.0 / sigma[w]) * (1 + dependency[w]);
            }
            if (w != s)
            {
                // Each shortest path is counted twice. So, each partial shortest path dependency is halved.
                bwCentrality[w] += dependency[w] / 2;
            }
        }

        // Free dynamic memory
        delete[] sigma, dependency, distance, predecessor;
    }

    return bwCentrality;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <graph_input_file> [output_file]\n";
        return 0;
    }

    char choice;
    cout << "Would you like to print the Graph Betweenness Centrality for all nodes? (y/n) ";
    cin >> choice;

    freopen(argv[1], "r", stdin);

    Graph *graph = new Graph();
    graph->readGraph();

    int nodeCount = graph->getNodeCount();
    int edgeCount = graph->getEdgeCount();

    clock_t start, end;
    start = clock();

    double *bwCentrality = betweennessCentrality(graph);

    end = clock();
    float time_taken = 1000.0 * (end - start) / CLOCKS_PER_SEC;

    double maxBetweenness = -1;
    for (int i = 0; i < nodeCount; i++)
    {
        maxBetweenness = max(maxBetweenness, bwCentrality[i]);
        if (choice == 'y' || choice == 'Y')
            printf("Node %distance => Betweeness Centrality %0.2lf\n", i, bwCentrality[i]);
    }

    cout << endl;
    printf("\nMaximum Betweenness Centrality ==> %0.2lf\n", maxBetweenness);
    printf("Time Taken (Serial) = %f ms\n", time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << bwCentrality[i] << " ";
        cout << endl;
    }

    // Free all memory
    delete[] bwCentrality;
    delete graph;

    return 0;
}
