#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <random>
#include <ctime>

using namespace std;

class EnvGame{
public:

    EnvGame(int _node_n){

        node_n = _node_n;
        travel_count = 0;
        cur_side = 0; // An engineer starts at A.
        group[0].resize(node_n); // connectivity among wires on building A
        group[1].resize(node_n); // connectivity among wires on building B
        lines[0].resize(node_n); // mapping from each endnode of building A to the corresponding endnode of building B
        lines[1].resize(node_n); // mapping from each endnode of building B to the corresponding endnode of building A

        clearWires( 0 );         // initializaiton
        clearWires( 1 );         // initializaiton
        randomGame();
  
    }

    ~EnvGame(){
    }

    void cleanAllWires(){
        // cleanAllWires
        // disconnect all installed wires at the current building and clean them
        clearWires( cur_side );

    }

    void reset(){
        // reset
        // make a new case for wire connection
        clearWires( 0 );
        clearWires( 1 );
        randomGame();
        travel_count = 0;
        cur_side = 0;

    }

    void connectWire(int node1, int node2){
        // connectWire
        // link two nodes of the current building with a wire.

        int team_node1 = group [cur_side] [node1];
        int team_node2 = group [cur_side] [node2];

        for(int i=0;i<node_n;i++)
            if ( group[cur_side][i] == team_node1 )    
                group[cur_side][i] = team_node2;

    }

    bool answer(vector<int> answer){
        // answer(vector<int> answer)
        // check whether your answer is correct.
        // answer [i] represents that endnode i of building A is connected to endnode answer[i] of building B.
        // if the answer is correct, return true
        // Otherwise, return false;

        int l = answer.size();

        if ( l != node_n)
            return false;
        else{
            for(int i=0;i<node_n;i++)
                if( answer [i] != lines[0][i] )
                    return false;
        }

        return true;

    }

    bool checkConnectivity ( int node1, int node2){
        // checkConnecivity
        // Think that the engineer makes a circuit that connects a lamp and a batterly serially while linking node1 and node2 on the current building. 
        // if light is on, return true
        // Otherwise, return false
        int c1, c2;
        c1 = lines[cur_side][node1];
        c2 = lines[cur_side][node2];
        
        int opp_side = 1 - cur_side;

        if ( group [opp_side][c1] == group [opp_side][c2] )
            return true;

        else return false;
    }

    void goToOtherSide (){
        // gotoOtherSide
        // go to the other building
        cur_side = 1 - cur_side;
        travel_count ++;
    }

    int getNodeN(){
        return node_n;
    }
    int getTravelCount(){
        return travel_count;
    }

private:
    
    int node_n;
    
    int cur_side; // 0 means being at building A. 1 means being at building B.
    int travel_count;
    vector<int> lines[2];// wires between A and B
    vector<int> group[2];

    void clearWires(int side){
        for(int i=0;i < node_n;i++)
            group [side][i] = i;
    }

    void randomGame(){
        srand ( time(NULL) );
        auto rng = std::default_random_engine {};

        for(int i=0;i<node_n;i++){
            lines[0][i] = i;
        }

        std::random_shuffle(std::begin(lines[0]), std::end(lines[0]));

        for(int i=0;i<node_n;i++){
            printf("%d ", lines[0][i]);
            lines[1][lines[0][i]] = i;      
        }
    }

};