// Advanced Software Practices I
// 20211584 Junyeong JANG
// described image : https://cdn.discordapp.com/attachments/1018473421098717239/1154388406097412096/image.png

#include <bits/stdc++.h>
using namespace std;

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int N;
    cin >> N;

    vector<int> dp(N + 7, 0x7FFFFFFF);
    vector<int> path(N + 7, -1);
    dp[1] = 0;
    path[1] = -1;
    dp[2] = 1;
    path[2] = 1;
    dp[3] = 2;
    path[3] = 2;
    dp[4] = 2;
    path[4] = 2;
    dp[5] = 3;
    path[5] = 4;
    dp[6] = 3;
    path[6] = 3;

    for (int i = 7; i <= N; i++)
    {
        int cur = i;
        for (int j = cur - 1;; j--)
        {
            int nxt_l = cur - j;
            int nxt_r = j;
            int mul_cnt;
            if (nxt_l > nxt_r)
                break;

            bool isMade = false;
            int par = nxt_r;
            while (par != -1)
            {
                if (par == nxt_l)
                {
                    isMade = true;
                    break;
                }
                par = path[par];
            }

            if (isMade)
                mul_cnt = dp[nxt_r] + 1;
            else
                mul_cnt = dp[nxt_l] + dp[nxt_r] + 1;

            if (dp[cur] >= mul_cnt)
            {
                dp[cur] = mul_cnt;
                path[cur] = nxt_r;
            }
        }
    }

    cout << dp[N] << ' ';

    int par = N;
    stack<int> paths;
    while (par != -1)
    {
        paths.push(par);
        par = path[par];
    }
    while (!paths.empty())
    {
        cout << paths.top() << ' ';
        paths.pop();
    }

    return 0;
}