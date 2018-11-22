/**
*   Generates randomized graph of specified size
*
*   @author Ashwin Joisa
*   @author Praveen Gupta
**/
//=============================================================================================//

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <graph_file_name>\n";
        return 0;
    }

    int n, m;
    int x, y;

    cout << "Enter number of nodes:";
    cin >> n;
    cout << "Enter number of edges:";
    cin >> m;

    freopen(argv[1], "w", stdout);
    cout << n << " " << m << endl;

    for (int i = 0; i < m; i++) {
        do {
            x = rand() % n;
            y = rand() % n;
        } while (x == y);

        cout << x << " " << y << endl;
    }
}
