#include "preprocess_accel.hpp"
#include <iostream>
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_headers.hpp"
#include "ap_int.h"

int main(){
	std::cout << "Starting Testbench...\n";
	//read image
    cv::Mat img = cv::imread("/home/timo/HW_Pre_Post/test_image.jpg",cv::IMREAD_COLOR);
    cv::Mat img_preprocessed(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
    if(img.empty()){
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }
    unsigned char scale = 255;
    std::cout << "Invoking Kernel..\n";
    preprocess_accel((ap_uint<PTR_WIDTH_IN> *)img.data,(ap_uint<PTR_WIDTH_OUT> *)img_preprocessed.data, scale);

    std::cout << "Saving Image..\n";
    cv::imwrite("preprocessed.jpg", img_preprocessed);
    std::cout << "Test finished\n";

	/*cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat img_preprocessed(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));

	float tracks[16] = {0};
	tracks[0] = 400;
	tracks[1] = 100;
	tracks[2] = 550;
	tracks[3] = 500;
	tracks[0+8] = 600;
	tracks[1+8] = 250;
	tracks[2+8] = 700;
	tracks[3+8] = 400;

	ap_uint<24> *color = (ap_uint<24>*)malloc(16 * sizeof(int));;
	color[0].range(7, 0) = 0;
	color[0].range(15, 8) = 0;
	color[0].range(23, 16) = 0;
	color[1].range(7, 0) = 255;
	color[1].range(15, 8) = 0;
	color[1].range(23, 16) = 0;

	preprocess_accel((ap_uint<PTR_WIDTH_IN>*)img.data,(ap_uint<PTR_WIDTH_IN>*)img_preprocessed.data, 0, (float *)tracks, 2, color, 1);

    std::cout << "Saving Image..\n";
    cv::imwrite("test.jpg", img);
    std::cout << "Test finished\n";*/

	return 0;
}
