#include "solve.hpp"

int solve(Party *party){

	// TODO

	// Auxiliary codes (can be deleted)
	bool AknowB, BknowA;
	int i=0, j=1;

	AknowB = party->askAToKnowB (i, j);
	BknowA = party->askAToKnowB (j, i);

	return -1;
}

