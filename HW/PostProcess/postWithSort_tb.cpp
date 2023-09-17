#include <fstream>
#include <iostream>
#include "postWithSort.hpp"
#include <cstdint>

int main(){

	static uint16_t bboxes_L[6];
	bboxes_L[0] = 116;
	bboxes_L[1] = 90;
	bboxes_L[2] = 156;
	bboxes_L[3] = 198;
	bboxes_L[4] = 373;
	bboxes_L[5] = 326;
	static uint16_t bboxes_M[6];
	bboxes_M[0] = 30;
	bboxes_M[1] = 61;
	bboxes_M[2] = 62;
	bboxes_M[3] = 45;
	bboxes_M[4] = 59;
	bboxes_M[5] = 119;
	static uint16_t bboxes_S[6];
	bboxes_S[0] = 10;
	bboxes_S[1] = 13;
	bboxes_S[2] = 16;
	bboxes_S[3] = 30;
	bboxes_S[4] = 33;
	bboxes_S[5] = 23;

	static int8_t results_L[2*2*3*8];
	static int8_t results_M[2*2*3*8];
	static int8_t results_S[2*2*3*8];

	float data_scale_L = 0;
	float data_scale_M = 0;
	float data_scale_S = 0;

	std::ifstream indata;
	float data;

	indata.open("/home/timo/HW_Pre_Post/postWithSort/results.txt");
	if(!indata) { // file couldn't be opened
	      std::cerr << "Error: file could not be opened" << std::endl;
	      return 1;
	}

	indata >> data;
	data_scale_L = data;
	indata >> data;
	data_scale_M = data;
	indata >> data;
	data_scale_S = data;
	indata.close();

	indata.open("/home/timo/HW_Pre_Post/postWithSort/results_L.txt");
	if(!indata) { // file couldn't be opened
	      std::cerr << "Error: file could not be opened" << std::endl;
	      return 1;
	}

	int counter = 0;
	int8_t data_L = 0;
	while(!indata.eof()){
		indata >> data_L;
		results_L[counter] = data_L;
		counter ++;
		if (counter >= 2*2*3*8){
			break;
		}
	}
	indata.close();

	indata.open("/home/timo/HW_Pre_Post/postWithSort/results_M.txt");
	if(!indata) { // file couldn't be opened
	      std::cerr << "Error: file could not be opened" << std::endl;
	      return 1;
	}

	counter = 0;
	int8_t data_M = 0;
	while(!indata.eof()){
		indata >> data_M;
		results_M[counter] = data_M;
		counter ++;
		if (counter >= 2*2*3*8){
			break;
		}
	}
	indata.close();

	indata.open("/home/timo/HW_Pre_Post/postWithSort/results_S.txt");
	if(!indata) { // file couldn't be opened
	      std::cerr << "Error: file could not be opened" << std::endl;
	      return 1;
	}

	counter = 0;
	int8_t data_S = 0;
	while(!indata.eof()){
		indata >> data_S;
		results_S[counter] = data_S;
		counter ++;
		if (counter >= 2*2*3*8){
			break;
		}
	}
	indata.close();

	int items_out = 0;
	static float box_out[10*8] = {0};

	std::cout << "Start\n";

	postWithSort(results_L, results_M, results_S, &items_out, box_out, bboxes_L, bboxes_M, bboxes_S, data_scale_L, data_scale_M, data_scale_S, 0.2);

	for (int i = 0; i < items_out; i++){
		std::cout << box_out[i*8] << " " << box_out[i*8+1] << " " << box_out[i*8+2] << " " << box_out[i*8+3] << " " << box_out[i*8+6] << std::endl;
	}

	return 0;
}
