#include <cstdio>
#include "party.hpp"
#include "solve.hpp"

int main(){
	Party party(10);
	party.setRandomCase();
	party.start();
	int guess = solve(&party);
	//system("ls");
	return party.answer(guess);	
}
