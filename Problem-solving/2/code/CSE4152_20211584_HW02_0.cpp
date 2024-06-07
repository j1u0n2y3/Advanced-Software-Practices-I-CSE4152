// Advanced Software Practices I
// 20211584 Junyeong JANG
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    int N;
    cin >> N;

    vector<vector<int>> arr(N);
    for (int i = 0; i < N; i++)
    {
        arr[i].resize(N);
        for (int j = 0; j < N; j++)
            cin >> arr[i][j];
    }

    vector<pair<int, int>> max_elem(N); // <val, idx>
    for (int i = 0; i < N; i++)
    {
        int idx = max_element(arr[i].begin(), arr[i].end()) - arr[i].begin();
        max_elem[i] = {arr[i][idx], idx};
    }

    for (int repeat = 0; repeat < N; repeat++)
    {
        int row = max_element(max_elem.begin(), max_elem.end()) - max_elem.begin();
        cout << max_elem[row].first << ' ';
        if (repeat == N - 1)
            break;
        arr[row][max_elem[row].second] = -0x7FFFFFFF;
        int nxt_idx = max_element(arr[row].begin(), arr[row].end()) - arr[row].begin();
        max_elem[row] = {arr[row][nxt_idx], nxt_idx};
    }

    return 0;
}