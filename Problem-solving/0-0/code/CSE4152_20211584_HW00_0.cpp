// Advanced Software Practices I
// 0-0
// 20211584 Junyeong JANG

#include <bits/stdc++.h>
using namespace std;

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    long long N;
    cin >> N;
    if (N == 0)
    {
        cout << 0;  return 0;
    }

    long long sum = 0, res = -9223372036854775806LL;
    for (int i = 0; i < N; i++)
    {
        int cur;    cin >> cur;

        sum = (sum < 0 ? 0 : sum) + cur;
        res = max(res, sum);
    }

    cout << (res < 0 ? 0 : res);

    return 0;
}