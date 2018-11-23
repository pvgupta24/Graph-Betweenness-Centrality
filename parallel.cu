/**
*   CUDA C/C++ implementation for Accelerating Graph Betweenness Centrality for Sparse Graphs
*
*   @author Ashwin Joisa
*   @author Praveen Gupta
**/

//=============================================================================================//

// Include header files
#include <iostream>
#include <cuda.h>

// Include custom header file for implementation of Graphs
#include "Graph.h"

//=============================================================================================//

#define MAX_THREAD_COUNT 1024
#define CEIL(a, b) ((a - 1) / b + 1)

//=============================================================================================//

using namespace std;

//=============================================================================================//

#define catchCudaError(error) { gpuAssert((error), __FILE__, __LINE__); }

// Catch Cuda errors
inline void gpuAssert(cudaError_t error, const char *file, int line,  bool abort = false)
{
    if (error != cudaSuccess)
    {
        printf("\n====== Cuda Error Code %i ======\n %s in CUDA %s\n", error, cudaGetErrorString(error));
        printf("\nIn file :%s\nOn line: %d", file, line);
        
        if(abort)
            exit(-1);
    }
}
//=============================================================================================//

__global__ void betweennessCentralityKernel(Graph *graph, double *bwCentrality, int nodeCount){

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
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

    return bwCentrality;
}

double *betweennessCentrality(Graph *graph, int nodeCount)
{
    double *bwCentrality = new double[nodeCount]();
    double *device_bwCentrality;

    //TODO: Allocate device memory for bwCentrality

    assignColoursKernel<<<CEIL(nodeCount, MAX_THREAD_COUNT), MAX_THREAD_COUNT>>>(graph, device_bwCentrality, nodeCount);
    cudaDeviceSynchronize();

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

    Graph *host_graph = new Graph();
    Graph *device_graph;

    catchCudaError(cudaMalloc((void **)&device_graph, sizeof(Graph)));
    host_graph->readGraph();

    int nodeCount = host_graph->getNodeCount();
    int edgeCount = host_graph->getEdgeCount();
    catchCudaError(cudaMemcpy(device_graph, host_graph, sizeof(Graph), cudaMemcpyHostToDevice));

    // Copy Adjancency List to device
    int *adjacencyList;
    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&adjacencyList, sizeof(int) * (2 * edgeCount + 1)));
    catchCudaError(cudaMemcpy(adjacencyList, host_graph->adjacencyList, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice));
    // Update the pointer to this, in device_graph
    catchCudaError(cudaMemcpy(&(device_graph->adjacencyList), &adjacencyList, sizeof(int *), cudaMemcpyHostToDevice));

    // Copy Adjancency List Pointers to device
    int *adjacencyListPointers;
    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&adjacencyListPointers, sizeof(int) * (nodeCount + 1)));
    catchCudaError(cudaMemcpy(adjacencyListPointers, host_graph->adjacencyListPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice));
    // Update the pointer to this, in device_graph
    catchCudaError(cudaMemcpy(&(device_graph->adjacencyListPointers), &adjacencyListPointers, sizeof(int *), cudaMemcpyHostToDevice));


    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start));
    catchCudaError(cudaEventCreate(&device_end));
    float device_time_taken;

    catchCudaError(cudaEventRecord(device_start));

    double *bwCentrality = betweennessCentrality(graph, nodeCount);

    catchCudaError(cudaEventRecord(device_end));
    catchCudaError(cudaEventSynchronize(device_end));
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    double maxBetweenness = -1;
    for (int i = 0; i < nodeCount; i++)
    {
        maxBetweenness = max(maxBetweenness, bwCentrality[i]);
        if (choice == 'y' || choice == 'Y')
            printf("Node %distance => Betweeness Centrality %0.2lf\n", i, bwCentrality[i]);
    }

    cout << endl;

    printf("\nMaximum Betweenness Centrality ==> %0.2lf\n", maxBetweenness);
    printf("Time Taken (Parallel) = %f ms\n", device_time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << bwCentrality[i] << " ";
        cout << endl;
    }

    // Free all memory
    delete[] colouring;
    catchCudaError(cudaFree(adjacencyList));
    catchCudaError(cudaFree(adjacencyListPointers));
    catchCudaError(cudaFree(device_graph));
}