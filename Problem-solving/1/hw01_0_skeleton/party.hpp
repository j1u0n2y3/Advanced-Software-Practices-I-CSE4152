#pragma once
#include <cstdio>
#include <vector>
#include <set>

class Party{
public:
	Party();
	Party(int n);
	void setTestCase( int case_n );
	void setRandomCase( );
	void start();
	bool askAToKnowB( int A, int B );
	int answer( int A );
	int getN();
private:
	int _prepared = 0;
	int _start = 0;
	int _n;
	int _celebrity;
	std::vector<std::set<int> > _relations;
	int _questionCount;
};
