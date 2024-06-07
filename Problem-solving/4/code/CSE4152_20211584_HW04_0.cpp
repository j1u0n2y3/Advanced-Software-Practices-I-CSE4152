// Advanced Software Practices I
// 20211584 Junyeong JANG
#include <iostream>
#include <stack>
using namespace std;

#define INF 0x7FFFFFFF

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    int T;
    cin >> T;
    while (T--)
    {
        int N;
        cin >> N;

        bool res = true;
        stack<int> rail;
        rail.push(INF);
        int waiting = 1;
        for (int i = 0; i < N; i++)
        {
            int cur;
            cin >> cur;
            if (cur < rail.top())
            {
                // cout << cur << " push to b. now waiting : " << waiting << '\n';
                rail.push(cur);
            }
            else
            {
                while (res && cur > rail.top() && rail.top() < INF)
                {
                    // cout << rail.top() << " pop from b. now waiting : " << waiting << '\n';
                    if (rail.top() != waiting)
                    {
                        res = false;
                        break;
                    }
                    rail.pop();
                    waiting++;
                }
                // cout << cur << " push to b. now waiting : " << waiting << '\n';
                rail.push(cur);
            }
        }
        while (res && rail.top() < INF)
        {
            // cout << rail.top() << " pop from b. now waiting : " << waiting << '\n';
            if (rail.top() != waiting)
            {
                res = false;
                break;
            }
            rail.pop();
            waiting++;
        }
        cout << (res ? "POSSIBLE\n" : "IMPOSSIBLE\n");
    }

    return 0;
}