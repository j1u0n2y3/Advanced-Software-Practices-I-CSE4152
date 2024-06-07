#include "party.hpp"
#include <cstdlib>
#include <ctime>
#include <cstdio>

Party::Party(){
}

void Party::setRandomCase(){

	int celebrity = rand() % _n;

	for(int i=0;i<_n;i++){

		_relations[i].clear();

		if (i == celebrity)
			continue;

		_relations[i].insert ( celebrity );

		for (int j=0;j<_n;j++){
			if (i!=j && rand() % 20 < 10)
				_relations[i].insert(j);
		}	
	}

	_celebrity = celebrity;
	_prepared = 1;

}

bool Party::askAToKnowB(int A, int B){
	if (!_start){
		printf("Please push the start button.\n");
		return false;
	}

	_questionCount += 1;
	bool cond =  ( _relations[A].find(B) != _relations[A].end() );
	if (cond) printf("%d knows %d.\n", A, B);
	else printf("%d doesn't know %d.\n", A, B);
	return cond;
}

int Party::getN(){
	return _n;
}

int Party::answer (int x){
	if (!_start){
		printf("Please push the start button.\n");
		return 1;
	}

	if ( x == _celebrity){
		printf("%d is the celebrity! You find the celebrity with %d questions.\n", x, _questionCount);
		return 0;
	}
	else{
		printf("Nope. Game end. Please set again.\n");
		_start = 0;
		return 1;
	}
}


void Party::start(){
	if ( _prepared ){
		_start = 1;
		_questionCount = 0;
	}
	else{
		printf("You should set the case first\n");
	}
}

Party::Party(int n){
	srand (time(NULL));
	_n = n;
	_relations.resize(n);
	_prepared = 0;
	_start = 0;
	_celebrity = -1;
	
}
