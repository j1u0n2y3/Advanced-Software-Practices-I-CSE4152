#include <bits/stdc++.h>
using namespace std;

int N, M, K, res = 0;
pair<int, int> S, E;
bool **graph;
int di[12] = {-1, 1, 0, 0, -1, 1, -1, 1, -2, 2, -2, 2},
    dj[12] = {0, 0, -1, 1, 2, 2, -2, -2, 1, 1, -1, -1};

void dfs(pair<int, int> cur, int depth)
{
    if (cur == E && depth == K)
    {
        res++;
        return;
    }
    else if ((cur == E && depth != K) || (cur != E && depth == K))
        return;

    for (int i = 0; i < 12; i++)
    {
        pair<int, int> nxt = {cur.first + di[i], cur.second + dj[i]};
        if (0 > nxt.first || nxt.first >= N || 0 > nxt.second || nxt.second >= M || graph[nxt.first][nxt.second] == 1)
            continue;

        dfs(nxt, depth + 1);
    }

    return;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    cin >> N >> M >> K;

    graph = new bool *[N];
    for (int i = 0; i < N; i++)
        graph[i] = new bool[M];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            char tmp;
            cin >> tmp;

            if (tmp == 'S')
            {
                S = {i, j};
                graph[i][j] = 0;
            }
            else if (tmp == 'E')
            {
                E = {i, j};
                graph[i][j] = 0;
            }
            else
                graph[i][j] = (int)(tmp - '0');
        }
    }

    dfs(S, 0);

    cout << res << '\n';

    for (int i = 0; i < N; i++)
        delete[] graph[i];
    delete[] graph;
    return 0;
}