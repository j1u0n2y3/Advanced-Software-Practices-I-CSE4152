// Advanced Software Practices I
// 20211584 Junyeong JANG
#include "EnvGame.h"

void leftNode(vector<int> &result, int nodeCount)
{
    vector<bool> temp(nodeCount, false);
    for (int i = 1; i < result.size(); i++)
    {
        temp[result[i]] = true;
    }
    for (int i = 0; i < nodeCount; i++)
    {
        if (!temp[i])
        {
            result[0] = i;
            return;
        }
    }
}

vector<int> solve(EnvGame &game)
{
    int nodeCount = game.getNodeN();
    vector<int> result(nodeCount);
    vector<vector<int>> connections(nodeCount);

    for (int i = 0; i < nodeCount; i += 2)
    {
        int nextNode = (i + 1) % nodeCount;

        game.cleanAllWires();
        game.connectWire(i, nextNode);
        game.goToOtherSide();

        for (int j = 0; j < nodeCount - 1; j++)
        {
            for (int k = j + 1; k < nodeCount; k++)
            {
                if (game.checkConnectivity(j, k))
                {
                    connections[nextNode].push_back(j);
                    connections[nextNode].push_back(k);
                    connections[i].push_back(j);
                    connections[i].push_back(k);
                }
            }
        }
        game.goToOtherSide();
    }

    for (int i = 1; i < nodeCount; i += 2)
    {
        int nextNode = (i + 1) % nodeCount;
        game.cleanAllWires();
        game.connectWire(i, nextNode);
        game.goToOtherSide();

        for (int j = 0; j < nodeCount - 1; j++)
        {
            for (int k = j + 1; k < nodeCount; k++)
            {
                if (game.checkConnectivity(j, k))
                {
                    auto it_ni_j = find(connections[nextNode].begin(), connections[nextNode].end(), j);
                    auto it_ni_k = find(connections[nextNode].begin(), connections[nextNode].end(), k);
                    auto it_i_j = find(connections[i].begin(), connections[i].end(), j);
                    auto it_i_k = find(connections[i].begin(), connections[i].end(), k);

                    if (it_i_j != connections[i].end())
                        result[i] = j;
                    else if (it_i_k != connections[i].end())
                        result[i] = k;

                    if (it_ni_j != connections[nextNode].end())
                        result[nextNode] = j;
                    else if (it_ni_k != connections[nextNode].end())
                        result[nextNode] = k;
                }
            }
        }
        game.goToOtherSide();
    }
    leftNode(result, nodeCount);

    return result;
}
