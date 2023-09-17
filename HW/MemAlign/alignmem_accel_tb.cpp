#include "alignmem_accel.hpp"
#include <iostream>
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_headers.hpp"
#include "ap_int.h"

int main(){
	std::cout << "Starting Testbench...\n";
	//read image
    cv::Mat img = cv::imread("/home/timo/HW_Pre_Post/test.jpg",cv::IMREAD_COLOR);
    cv::Mat img_preprocessed(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
    if(img.empty()){
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    std::cout << "Invoking Kernel..\n";
    alignmem_accel((ap_uint<128> *)img.data,(ap_uint<128> *)img_preprocessed.data);

    std::cout << "Saving Image..\n";
    cv::imwrite("preprocessed.jpg", img_preprocessed);
    std::cout << "Test finished\n";

    return 0;
}
