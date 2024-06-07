// Advanced Software Practices I
// 20211584 Junyeong JANG

#include "solve.hpp"

#ifndef NO_CELEB
int solve(Party *party)
{
    int N = party->getN();

    int lp = 0, rp = N - 1;
    while (lp < rp)
    {
        bool answer = party->askAToKnowB(lp, rp);

        if (answer == true)
            lp++;
        else
            rp--;
    }

    return lp;
}
#else
int solve(Party *party)
{
    int N = party->getN();

    queue<int> Q;
    for (int i = 0; i < N; i++)
        Q.push(i);

    map<pair<int, int>, bool> questions;
    while (Q.size() > 1)
    {
        int qsize = Q.size();
        while (qsize)
        {

            int a = Q.front();
            Q.pop();
            qsize--;
            if (Q.empty())
            {
                Q.push(a);
                break;
            }
            int b = Q.front();
            Q.pop();
            qsize--;

            if (questions[make_pair(a, b)] = party->askAToKnowB(a, b))
                Q.push(b);
            else
                Q.push(a);
        }
    }

    int candidate = Q.front();
    Q.pop();
    for (int i = 0; i < N; i++)
    {
        if (i == candidate)
            continue;

        map<pair<int, int>, bool>::iterator ans = questions.find(make_pair(i, candidate));
        if (ans != questions.end())
        {
            if ((*ans).second == false)
                return -1;
            else
                continue;
        }

        if (party->askAToKnowB(i, candidate) == false)
            return -1;
    }

    for (int i = 0; i < N; i++)
    {
        if (i == candidate)
            continue;

        map<pair<int, int>, bool>::iterator ans = questions.find(make_pair(candidate, i));
        if (ans != questions.end())
        {
            if ((*ans).second == true)
                return -1;
            else
                continue;
        }

        if (party->askAToKnowB(candidate, i) == true)
            return -1;
    }

    return candidate;
}
#endif /* NO_CELEB */