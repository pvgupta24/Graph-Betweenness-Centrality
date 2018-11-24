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

// Max device memory : 4 GB
#define MAX_MEMORY ((long long)4e9)

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
    printf("Time Taken in milliseconds : %d\n", (int)ms);
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

__global__ void betweennessCentralityKernel(Graph *graph, float *bwCentrality, int nodeCount,
            int *sigma, int *distance, float *dependency, int *Q, int *Qpointers) {
    
    int idx = threadIdx.x;
    if(idx >= nodeCount)
        return;
    
    __shared__ int s;
    __shared__ int Q_len;
    __shared__ int Qpointers_len;
    __shared__ int noOfBlocks;

    if(idx == 0) {
        s = blockIdx.x - gridDim.x;
        noOfBlocks = gridDim.x;
        // printf("Progress... %3d%%", 0);
    }
    __syncthreads();
    
    while(s < nodeCount - noOfBlocks)
    {
        if(idx == 0)
        {
            s += noOfBlocks;
            // printf("\rProgress... %5.2f%%", (s+1)*100.0/nodeCount);
            // printf("Node %d\n", s);
            
            Q[0 + (blockIdx.x * nodeCount)] = s;
            Q_len = 1;
            Qpointers[0 + (blockIdx.x * nodeCount)] = 0;
            Qpointers[1 + (blockIdx.x * nodeCount)] = 1;
            Qpointers_len = 1;
        }
        __syncthreads();

        for(int v=idx; v<nodeCount; v+=blockDim.x)
        {
            if(v == s)
            {
                distance[v + (blockIdx.x * nodeCount)] = 0;
                sigma[v + (blockIdx.x * nodeCount)] = 1;
            }
            else
            {
                distance[v + (blockIdx.x * nodeCount)] = INT_MAX;
                sigma[v + (blockIdx.x * nodeCount)] = 0;
            }
            dependency[v + (blockIdx.x * nodeCount)] = 0.0;
        }
        __syncthreads();
        
        // BFS
        while(true)
        {
            __syncthreads();
            for(int k=idx; k<Qpointers[Qpointers_len + (blockIdx.x * nodeCount)]; k+=blockDim.x) 
            {
                if(k < Qpointers[Qpointers_len -1 + (blockIdx.x * nodeCount)])
                    continue;

                int v = Q[k + (blockIdx.x * nodeCount)];
                for(int r = graph->adjacencyListPointers[v]; r < graph->adjacencyListPointers[v + 1]; r++)
                {
                    int w = graph->adjacencyList[r];
                    if(atomicCAS(&distance[w + (blockIdx.x * nodeCount)], INT_MAX, distance[v + (blockIdx.x * nodeCount)] +1) == INT_MAX)
                    {
                        int t = atomicAdd(&Q_len, 1);
                        Q[t + (blockIdx.x * nodeCount)] = w;
                    }
                    if(distance[w + (blockIdx.x * nodeCount)] == (distance[v + (blockIdx.x * nodeCount)]+1))
                    {
                        atomicAdd(&sigma[w + (blockIdx.x * nodeCount)], sigma[v + (blockIdx.x * nodeCount)]);
                    }
                }
            }
            __syncthreads();

            if(Q_len == Qpointers[Qpointers_len + (blockIdx.x * nodeCount)])
                break;

            if(idx == 0)
            {
                Qpointers_len++;
                Qpointers[Qpointers_len + (blockIdx.x * nodeCount)] = Q_len;
            }
            __syncthreads();
        }
        __syncthreads();
        
        // Reverse BFS
        while(Qpointers_len > 0)
        {
            for(int k=idx; k < Qpointers[Qpointers_len + (blockIdx.x * nodeCount)]; k+=blockDim.x) 
            {
                if(k < Qpointers[Qpointers_len -1 + (blockIdx.x * nodeCount)])
                    continue;

                int v = Q[k + (blockIdx.x * nodeCount)];
                for(int r = graph->adjacencyListPointers[v]; r < graph->adjacencyListPointers[v + 1]; r++)
                {
                    int w = graph->adjacencyList[r];
                    if(distance[w + (blockIdx.x * nodeCount)] == (distance[v + (blockIdx.x * nodeCount)] + 1))
                    {
                        if (sigma[w + (blockIdx.x * nodeCount)] != 0)
                            dependency[v + (blockIdx.x * nodeCount)] += (sigma[v + (blockIdx.x * nodeCount)] * 1.0 / sigma[w + (blockIdx.x * nodeCount)]) * (1 + dependency[w + (blockIdx.x * nodeCount)]);
                    }
                }
                if (v != s)
                {
                    // Each shortest path is counted twice. So, each partial shortest path dependency is halved.
                    atomicAdd(bwCentrality + v, dependency[v + (blockIdx.x * nodeCount)] / 2);
                }
            }
            __syncthreads();

            if(idx == 0)
                Qpointers_len--;

            __syncthreads();
        }
    }
}

float *betweennessCentrality(Graph *graph, int nodeCount)
{
    float *bwCentrality = new float[nodeCount]();
    float *device_bwCentrality, *dependency;
    int *sigma, *distance, *Q, *Qpointers;

    const int BLOCK_COUNT = MAX_MEMORY / (4 * 5 * nodeCount);
    // pritnf(">> %d\n", BLOCK_COUNT);

    //TODO: Allocate device memory for bwCentrality
    catchCudaError(cudaMalloc((void **)&device_bwCentrality, sizeof(float) * nodeCount));
    catchCudaError(cudaMalloc((void **)&sigma, sizeof(int) * nodeCount * BLOCK_COUNT));
    catchCudaError(cudaMalloc((void **)&distance, sizeof(int) * nodeCount * BLOCK_COUNT));
    catchCudaError(cudaMalloc((void **)&Q, sizeof(int) * (nodeCount) * BLOCK_COUNT));
    catchCudaError(cudaMalloc((void **)&Qpointers, sizeof(int) * (nodeCount) * BLOCK_COUNT));
    catchCudaError(cudaMalloc((void **)&dependency, sizeof(float) * nodeCount * BLOCK_COUNT));

    catchCudaError(cudaMemcpy(device_bwCentrality, bwCentrality, sizeof(float) * nodeCount, cudaMemcpyHostToDevice));

    // Timer
    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start));
    catchCudaError(cudaEventCreate(&device_end));
    catchCudaError(cudaEventRecord(device_start));
    

    betweennessCentralityKernel<<<BLOCK_COUNT, MAX_THREAD_COUNT>>>(graph, device_bwCentrality, nodeCount, sigma, distance, dependency, Q, Qpointers);
    cudaDeviceSynchronize();
    //End of progress bar
    cout << endl;

    // Timer
    catchCudaError(cudaEventRecord(device_end));
    catchCudaError(cudaEventSynchronize(device_end));
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    // Copy back and free memory
    catchCudaError(cudaMemcpy(bwCentrality, device_bwCentrality, sizeof(float) * nodeCount, cudaMemcpyDeviceToHost));
    catchCudaError(cudaFree(device_bwCentrality));
    catchCudaError(cudaFree(sigma));
    catchCudaError(cudaFree(dependency));
    catchCudaError(cudaFree(distance));
    catchCudaError(cudaFree(Q));
    catchCudaError(cudaFree(Qpointers));
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

    float *bwCentrality = betweennessCentrality(device_graph, nodeCount);

    float maxBetweenness = -1;
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