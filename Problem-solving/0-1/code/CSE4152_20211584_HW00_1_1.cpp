#include <bits/stdc++.h>
using namespace std;

struct Coin
{
    int val;
    int cnt;

    bool operator<(const Coin &c) const
    {
        return val < c.val;
    }
};

bool length_comp(string a, string b)
{
    return a.length() < b.length();
}

int N, S;
vector<Coin> coins;
stack<int> comb_stack;
vector<string> output;

void dfs(int sum, int iter)
{
    if (sum > S || iter == N)
        return;
    else if (sum == S)
    {
        stack<int> _comb_stack = comb_stack;
        string comb = "";
        while (!_comb_stack.empty())
        {
            comb.append(to_string(_comb_stack.top()));
            comb.push_back(' ');
            _comb_stack.pop();
        }
        output.push_back(comb);
        return;
    }

    for (int i = iter; i < N; i++)
    {
        if (coins[i].cnt == 0)
            continue;

        comb_stack.push(coins[i].val);
        coins[i].cnt--;
        /***/
        dfs(sum + coins[i].val, i);
        /***/
        coins[i].cnt++;
        comb_stack.pop();
    }

    return;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    cin >> N >> S;

    coins.resize(N);
    for (int i = 0; i < N; i++)
        cin >> coins[i].val >> coins[i].cnt;
    sort(coins.begin(), coins.end());

    dfs(0, 0);
    sort(output.begin(), output.end(), length_comp);

    cout << output.size() << '\n';
    for (int i = 0; i < output.size(); i++)
        cout << output[i] << '\n';

    return 0;
}