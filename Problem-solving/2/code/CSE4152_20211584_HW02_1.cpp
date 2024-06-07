// Advanced Software Practices I
// 20211584 Junyeong JANG
#include <iostream>
using namespace std;

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    int N, **mat;
    cin >> N;
    mat = new int *[N];
    for (int i = 0; i < N; i++)
        mat[i] = new int[N];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cin >> mat[i][j];

    int K;
    cin >> K;
    int ip = N - 1, jp = 0;
    while (ip >= 0 && jp < N)
    {
        int cur = mat[ip][jp];
        if (cur == K)
        {
            cout << ip + 1 << ' ' << jp + 1 << '\n';
            return 0;
        }
        else if (cur > K)
            ip--;
        else if (cur < K)
            jp++;
    }
    cout << "Loop end. There is no " << K << " in this matrix.\n";

    return 0;
}