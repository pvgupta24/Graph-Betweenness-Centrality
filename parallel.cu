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

float device_time_taken;

void printTime(float ms) {
    int h = ms / (1000*3600);
    int m = (((int)ms) / (1000*60)) % 60;
    int s = (((int)ms) / 1000) % 60;
    int intMS = ms;
    intMS %= 1000;

    printf("Time Taken (Parallel) = %dh %dm %ds %dms\n", h, m, s, intMS);
    printf("Time Taken in milliseconds : %d\n", intMS);
}

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

__global__ void betweennessCentralityKernel(Graph *graph, double *bwCentrality, int nodeCount,
            int *sigma, int *distance, double *dependency) {
    
    int idx = threadIdx.x;
    if(idx >= nodeCount)
        return;
    
    __shared__ int s;
    __shared__ int current_depth;
    __shared__ bool done;

    if(idx == 0) {
        s = -1;
        printf("Progress... %3d%%", 0);
    }
    __syncthreads();

    while(s < nodeCount)
    {    
        if(idx == 0)
        {
            ++s;
            printf("\rProgress... %5.2f%%", (s+1)*100.0/nodeCount);
            done = false;
            current_depth = -1;
        }
        __syncthreads();

        //Initialize distance and sigma
        for(int v=idx; v<nodeCount; v+=blockDim.x)
        {
            if(v == s)
            {
                distance[v] = 0;
                sigma[v] = 1;
            }
            else
            {
                distance[v] = INT_MAX;
                sigma[v] = 0;
            }
            dependency[v] = 0.0;
        }
        __syncthreads();
        
        // Calculate the number of shortest paths and the 
        // distance from s (the root) to each vertex
            
        while(!done)
        {
            if(idx == 0){
                current_depth++;
            }
            done = true;
            __syncthreads();

            for(int v=idx; v<nodeCount; v+=blockDim.x) //For each vertex...
            {
                if(distance[v] == current_depth)
                {
                    for(int r = graph->adjacencyListPointers[v]; r < graph->adjacencyListPointers[v + 1]; r++)
                    {
                        int w = graph->adjacencyList[r];
                        if(distance[w] == INT_MAX)
                        {
                            distance[w] = distance[v] + 1;
                            done = false;
                        }
                        if(distance[w] == (distance[v] + 1))
                        {
                            atomicAdd(&sigma[w], sigma[v]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Reverse BFS
        while(current_depth)
        {
            if(idx == 0){
                current_depth--;
            }
            __syncthreads();

            for(int v=idx; v<nodeCount; v+=blockDim.x) //For each vertex...
            {
                if(distance[v] == current_depth)
                {
                    for(int r = graph->adjacencyListPointers[v]; r < graph->adjacencyListPointers[v + 1]; r++)
                    {
                        int w = graph->adjacencyList[r];
                        if(distance[w] == (distance[v] + 1))
                        {
                            if (sigma[w] != 0)
                                dependency[v] += (sigma[v] * 1.0 / sigma[w]) * (1 + dependency[w]);
                        }
                    }
                    if (v != s)
                    {
                        // Each shortest path is counted twice. So, each partial shortest path dependency is halved.
                        bwCentrality[v] += dependency[v] / 2;
                    }
                }
            }
            __syncthreads();
        }
    }
}

double *betweennessCentrality(Graph *graph, int nodeCount)
{
    double *bwCentrality = new double[nodeCount]();
    double *device_bwCentrality, *dependency;
    int *sigma, *distance;

    //TODO: Allocate device memory for bwCentrality
    catchCudaError(cudaMalloc((void **)&device_bwCentrality, sizeof(double) * nodeCount));
    catchCudaError(cudaMalloc((void **)&sigma, sizeof(int) * nodeCount));
    catchCudaError(cudaMalloc((void **)&distance, sizeof(int) * nodeCount));
    catchCudaError(cudaMalloc((void **)&dependency, sizeof(double) * nodeCount));
    catchCudaError(cudaMemcpy(device_bwCentrality, bwCentrality, sizeof(double) * nodeCount, cudaMemcpyHostToDevice));

    // Timer
    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start));
    catchCudaError(cudaEventCreate(&device_end));
    catchCudaError(cudaEventRecord(device_start));

    betweennessCentralityKernel<<<1, MAX_THREAD_COUNT>>>(graph, device_bwCentrality, nodeCount, sigma, distance, dependency);
    cudaDeviceSynchronize();
    //End of progress bar
    cout << endl;

    // Timer
    catchCudaError(cudaEventRecord(device_end));
    catchCudaError(cudaEventSynchronize(device_end));
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    // Copy back and free memory
    catchCudaError(cudaMemcpy(bwCentrality, device_bwCentrality, sizeof(double) * nodeCount, cudaMemcpyDeviceToHost));
    catchCudaError(cudaFree(device_bwCentrality));
    catchCudaError(cudaFree(sigma));
    catchCudaError(cudaFree(dependency));
    catchCudaError(cudaFree(distance));
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

    double *bwCentrality = betweennessCentrality(device_graph, nodeCount);

    double maxBetweenness = -1;
    for (int i = 0; i < nodeCount; i++)
    {
        maxBetweenness = max(maxBetweenness, bwCentrality[i]);
        if (choice == 'y' || choice == 'Y')
            printf("Node %d => Betweeness Centrality %0.2lf\n", i, bwCentrality[i]);
    }

    cout << endl;

    printf("\nMaximum Betweenness Centrality ==> %0.2lf\n", maxBetweenness);
    printTime(device_time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << bwCentrality[i] << " ";
        cout << endl;
    }

    // Free all memory
    delete[] bwCentrality;
    catchCudaError(cudaFree(adjacencyList));
    catchCudaError(cudaFree(adjacencyListPointers));
    catchCudaError(cudaFree(device_graph));
}