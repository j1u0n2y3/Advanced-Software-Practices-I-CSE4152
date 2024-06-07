// Advanced Software Practices I
// 20211584 Junyeong JANG
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Point
{
    double x, y;
};

double dist(Point a, Point b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

int CCW(Point a, Point b, Point c)
{
    double ccw = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    if (ccw > 0)
        return 1;
    if (ccw < 0)
        return -1;
    return 0;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    int N;
    cin >> N;
    if (N < 3)
    {
        cout << "HCP";
        return 0;
    }

    vector<Point> points(N);
    Point O = {0, 0};
    for (int i = 0; i < N; i++)
        cin >> points[i].x >> points[i].y;
    if (CCW(points[0], points[1], O) == 0)
        swap(points[1], points[N - 1]);

    Point src = points[0], dst = points[1];
    int ccw = CCW(src, dst, O);

    for (int i = 2; i < N; i++)
    {
        Point p = points[i];
        if (CCW(src, dst, p) != ccw)
            continue;

        (dist(p, src) < dist(p, dst)) ? src = p : dst = p;

        int c = CCW(src, dst, O);
        if (c == 0)
            continue;
        if (c != ccw)
        {
            cout << "NO HCP";
            return 0;
        }
    }
    cout << "HCP";
    return 0;
}