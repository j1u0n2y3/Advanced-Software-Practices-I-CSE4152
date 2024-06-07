// Advanced Software Practices I
// 20211584 Junyeong JANG
#include <iostream>
#include <vector>
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
        arr[i].resize(N + 1, 0);

    for (int i = 0; i < N; i++)
    {
        int r1, r2, c1, c2, v;
        cin >> r1 >> r2 >> c1 >> c2 >> v;

        r1--;
        r2--;
        c1--;
        c2--;
        for (int j = r1; j <= r2; j++)
        {
            arr[j][c1] += v;
            arr[j][c2 + 1] += -v;
        }
    }
    for (int i = 0; i < N; i++)
        for (int j = 1; j < N; j++)
            arr[i][j] = arr[i][j - 1] + arr[i][j];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << arr[i][j] << ' ';
        cout << '\n';
    }

    return 0;
}