#include <iostream>
#include "common.hpp"
#include <cstdint>
#include <cmath>
#include <chrono> 

/**
 * @brief main 
 *
 * main function that calls the program loop  
 *
 * @param *argv[in]: commands to execute the program 
 *  
 */
int main(const int argc, const char **argv) {
    //call main program loop  
    int status = runDPU((int)argv[1][0]-48,(int)argv[2][0]-48,argv[3],argv[4],argv[5],argv[6]);
    std::cout << "Program terminated\n";
    return status; 
}