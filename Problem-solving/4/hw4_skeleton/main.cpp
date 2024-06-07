#include <cstdio>

#include "EnvGame.h"

vector<int> solve(EnvGame &x);

int main(){
    int n = 999;
    EnvGame a(n);

    vector<int> ans = solve(a);

    if( a.answer(ans))
        printf("Right. You solve it with %d travels", a.getTravelCount() );
    else printf("Wrong.");

    return 0;
}