# Graph Betweenness Centrality for sparse graphs with CUDA C/C++

- [Ashwin Joisa](https://github.com/AJ163)
- [Praveen Kumar Gupta](https://github.com/pvgupta24)

## Introduction

Betweenness Centrality is a measureof the influence of a vertex in a graph.
It measures the ratio of shortest paths passing through a particular vertex
to the total number of shortest paths between all pairs of vertices.
Intuitively, this ratio determines how well a vertex connects pairs of
vertices in the network.

Mathematically, the Betweenness Centrality of a vertex v is defined as:

![BC Equation](https://s0.wp.com/latex.php?zoom=1.2999999523162842&latex=BC%28v%29+%3D+%5Csum_%7Bs+%5Cneq+t+%5Cneq+v%7D+%5Cfrac%7B%5Csigma_%7Bst%7D%28v%29%7D%7B%5Csigma_%7Bst%7D%7D&bg=ffffff&fg=000&s=0)

Some of the use cases of Betweenness centrality includes finding the best
locations for stores within cities, power grid contingency analysis, and
community detection.

<hr>

## Graph Generation and formats
The adjacency matrix representation of storing Graphs take memory of the order
of O(V<sup>2</sup>).

Instead of the conventional way of storing graphs as adjacency matrix for sparse graphs,
i.e. graphs where the number of edges is of the order of O(V), there are other memory
efficient ways to represent and store them.

### CSR format
Compressed Sparse Row (CSR) format is a common way to store Graphs for GPU computations.
It consists of 2 arrays:
- Adjacency List array (Column Indices): The Adjacency List array is a concatenation
of each vertex’s adjacency
list into an array of E elements.
- Adjacency List Pointers array (Row Offset): Array of V+1 element that points
at where each vertex’s adjacency list begins and ends within the column indices
array.

For example, the adjacency list of vertex `s` starts at `adjPointers[adj[s]]` and
ends at `adjPointers[adj[s+1]-1]` inclusively.

### COO format
Cooperative (COO) format is another commonly used format which is essentially a
list of edges. Corresponding elements in 2 arrays `edgeList1` and `edgeList2` of
length E represent the head and tail vertices of each edge.

For undirected, we treat each undirected edge as two directed edges.
Adjacency List array of CSR format, is the same as `edgeList2` in COO format.

<hr>

## Brande's Algorithm
The naive implementation of solving the all-pairs shortest paths problem is of
the order of O(n<sup>3</sup>) by Floyd-Warshall technique and Betweeness Centrality
is calculated by counting the necessary shortest paths.

Brande's algorithm aims to reuse the already calculated shortest distances by using
partial dependecies of shortest paths between pairs of nodes for calculating BC
of a given vertex w.r.t a fixed root vertex.

A recursive relation for defining the partial dependency(𝛿) w.r.t root node s is:

![Brande's Partial Dependency equation](https://s0.wp.com/latex.php?zoom=1.2999999523162842&latex=%5Cdelta_s%28v%29+%3D+%5Csum_%7Bw%3Av+%5Cin+pred%28w%29%7D+%5Cfrac%7B%5Csigma_%7Bsv%7D%7D%7B%5Csigma_%7Bsw%7D%7D%281+%2B+%5Cdelta_s%28w%29%29&bg=ffffff&fg=000&s=0)

Effectively the calculation of BC values can then be divided into 2 steps:

1. Calculating the partial dependencies w.r.t all nodes
This is done by fixing a root and doing a forward Breadth First Search to calculate the depth of
other nodes w.r.t the fixed root s

2. Accumulating the partial dependencies for all root nodes to calculate the BC value
This is done using a reverse Breadth First Search by level-order traversal from the deepest level
towards the root to calculate partial dependencies of the shortest paths between
the current node to the fixed root which passes for all the predecessor nodes,
i.e. the nodes which are immediately one level above of the current node and the shortest
path from root to the current elements passes through them.

This gives a running Time of O(V*E)

We have implemented the following algorithms

1. Serial Implementation
2. Parallel Implementation
    - Vertex based
    - Edge based
    - Work efficient method (fine-grained parallelism)
    - Work distributed method with coarse-grained parallelism

- Multiple blocks in a CUDA grid were used for parallelising partial dependencies for indpendent roots. This helped us achieve a coarse grained parallelism by processing each source independently in parallel. Fine grained parallelism was achieved by running a parallel BFS using optimal work distribution strategies.
- Multi-sourced bottom up Breadth First Search for calculating dependency values of all nodes for a given source. A top down BFS was used to calculate the distance and number of shortest paths to each node with respect to a given source. Using these values, dependency values can were calculated by traversing the graph bottom up, using a multi-sourced bottom up Breadth First Search in parallel.

<hr>

## Running Instructions
All the implementations can take in graphs in CSR formats and optionally store
the Betweenness Centrality in an output file.

### Generating Graphs
Make the necessary directory for storing the generated graphs:

`mkdir testcases && mkdir testcases/1x`

Compile the random_graph_generator which generates and stores the graph file in
CSR format.

`g++ random_graph_generator.cpp -o _generate -std=c++11`

Run the binary and specify the number of nodes and edges needed in the graph:
` ./_generate testcases/1x/1e3.in`

> Enter number of nodes: 1000 <br>
Enter number of edges: 1000


> Note: Its optional to mention an output file, in which case the individual
Betweeness Centrality of vertices won't be stored

### Serial Implementation
Compile the serial BC implementation using

`g++ serial.cpp -o _serial -std=c++11`

Run the generated binary against the graphs generated in CSR formats as:

`./_serial testcases/1x/1e3.in testcases/1x/1e3s.out`

### Parallel Implementation
1. Vertex based

    Compile and Run using

    `nvcc parallel-vertex.cpp -o _parallel_vertex -std=c++11`

    `./_parallel_vertex testcases/1x/1e3.in testcases/1x/1e3pv.out`

2. Edge based

    Compile and Run using

    `nvcc parallel-edge.cpp -o _parallel_edge -std=c++11`

    `./_parallel_edge testcases/1x/1e3.in testcases/1x/1e3pe.out`

3. Work efficient method (fine-grained parallelism)

    Compile and Run using

    `nvcc parallel-work.cpp -o _parallel_work -std=c++11`

    `./_parallel-work testcases/1x/1e3.in testcases/1x/1e3pw.out`

4. Work distributed method with coarse-grained parallelism

    Compile and Run using

    `nvcc parallel-work-coarse.cpp -o _parallel_work_coarse -std=c++11`

    `./_parallel_work_coarse testcases/1x/1e3.in testcases/1x/1e3pwc.out`

> Note: Amount of blocks running in parallel can be controlled by changing the maximize
GPU memory to be allocated in MAX_MEMORY. By default it uses a maximum of 4GB.

<hr>

## Results and Analysis
The implementations were tested and compared on the Nvidia P100 GPU.

For the following given example graph:

![Example Graph](https://drive.google.com/uc?id=1svIaPGkhNDE14zHpQ5WmtuqRCSlyoPF7)

Output:

![Example Graph Output](https://drive.google.com/uc?id=1eYx4Q5POnx-w7nlvlqRaXjaGyCQSr97t)

<hr>

### Execution Time
|Number of Vertices	|Serial	|Parallel Vertex	|Parallel Edge	|Parallel Work Efficient	|Parallel Work Efficient with Coarse grained Parallelism	|Max BC|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|10<sup>3</sup>	|160	|198	|112	|78	|6	|27201.96|
|10<sup>4</sup>	|15430	|11703	|7665	|3749	|347	|1136948.18|
|10<sup>5</sup>	|2124190	|1409466	|825454	|279611	|37931	|13591397.85|


![Running Time Comparision](https://drive.google.com/uc?id=1YSd5Lv1EMcV2EgkeyqxNYTUd4YBQ42Eq)

<hr>

### Speed up with respect to Serial execution
|Number of Vertices	|Vertex Parallel	|Edge Parallel 	|Work Efficient	|Work Efficient with Coarse grained Parallelism	|Max BC|
|:---:|:---:|:---:|:---:|:---:|:---:|
|10<sup>3</sup>	|0.81|	1.43	|2.05	|26.67	|27201.96|
|10<sup>4</sup>	|1.32	|2.01	|4.12	|44.47	|1136948.18|
|10<sup>5</sup>	|1.51	|2.57	|7.60	|56.00	|13591397.85|

![Speedup Comparision](https://drive.google.com/uc?id=1O2EHJ46wh4gAAW33Z84z25t4lfhoFkf1)

<hr>

## References
- https://devblogs.nvidia.com/accelerating-graph-betweenness-centrality-cuda/
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.728.2926&rep=rep1&type=pdf
- http://matteo.rionda.to/centrtutorial/BonchiDeFrancisciMoralesRiondato-CentralityBigGraphsTutorial-Slides.pdf
