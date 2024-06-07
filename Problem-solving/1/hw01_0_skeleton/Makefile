
party.o: party.cpp
	g++ party.cpp -c -o party.o
solve.o: solve.cpp
	g++ solve.cpp -c -o solve.o
test: main.cpp solve.o party.o
	g++ main.cpp solve.o party.o -o test
