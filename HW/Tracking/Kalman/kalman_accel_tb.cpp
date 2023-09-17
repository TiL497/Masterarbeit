#include "kalman_accel.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_headers.hpp"

#define DIM 8
#define DT 0.5f
#define VAR_ACC std::pow(5,2)
#define VAR_X   std::pow(0.1,2)
#define VAR_Y   std::pow(0.1,2)
#define VAR_W   std::pow(0.001,2)
#define VAR_H   std::pow(0.001,2)

int main(){

    //cv::namedWindow("Kalman C++ Test");

    unsigned char predict_flag = 1 + 2 + 8 + 16;
    unsigned char update_flag = 1 + 4 + 32 + 64;

	float box_width = 50;
	float box_height = 100;
	float width = 600;
	float height = 600;
	float x0 = width/2;
	float y0 = height/2;
	float x = x0;
	float y = y0;
	float vx = 0;
	float vy = 0;
	float vw = 0;
	float vh = 0;
	float t = 0.5;

	int counter = 0;

	//State Vector
	float X[DIM*2] = {x0, y0, box_width, box_height, 0, 0, 0, 0, x0, y0, box_width, box_height, 0, 0, 0, 0};

    float A[DIM*DIM] = {1 ,0 ,0 ,0 ,DT,0 ,0 ,0 ,
                        0 ,1 ,0 ,0 ,0 ,DT,0 ,0 ,
                        0 ,0 ,1 ,0 ,0 ,0 ,DT,0 ,
                        0 ,0 ,0 ,1 ,0 ,0 ,0 ,DT,
                        0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 };
    //Define Measurement Mapping Matrix
    float H[4*DIM]   = {1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 };
    float Q[DIM*DIM] = {std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,
                        0                        ,std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,
                        0                        ,0                        ,std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,
                        0                        ,0                        ,0                        ,std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,
                        std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   ,0                        ,0                        ,0                        ,
                        0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   ,0                        ,0                        ,
                        0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   ,0                        ,
                        0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   };
	float Dq[DIM]    = {std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,2)*VAR_ACC ,std::pow(DT,2)*VAR_ACC ,std::pow(DT,2)*VAR_ACC ,std::pow(DT,2)*VAR_ACC};
    //Initial Measurement Noise Covariance
    float R[4*4]     = {VAR_X ,0     ,0     ,0    ,
                        0     ,VAR_Y ,0     ,0    ,
                        0     ,0     ,VAR_W ,0    ,
                        0     ,0     ,0     ,VAR_H};
    float Dr[4]      = {VAR_X ,VAR_Y ,VAR_W ,VAR_H};
	//Initial Covariance Matrix
	float Up[DIM*DIM*2] = {1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
						0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,
						0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,
						0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,
						0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,
						0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,
						0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
						0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,
						1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
						0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,
						0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,
						0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,
						0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,
						0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,
						0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
						0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 };
	float Dp[DIM*2] = {1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,1 ,1 ,1};

	float measure[4*2] = {0, 0, 0, 0, 0, 0, 0, 0};

	bool is_active[2] = {true, true};

	while (true){

		std::random_device rd; // obtain a random number from hardware
		std::mt19937 gen(rd()); // seed the generator
		std::uniform_int_distribution<> distr1(-5, 5); // define the range
		std::uniform_int_distribution<> distr2(-10, 20); // define the range
		//random acceleration
		float ax = float(distr1(gen))/10.0;
		float ay = float(distr1(gen))/10.0;
		float aw = float(distr2(gen))/100000.0;
		float ah = float(distr2(gen))/100000.0;
		//velocities
		vx = vx + ax * t;
		vy = vy + ay * t;
		vw = vw + aw * t;
		vh = vh + ah * t;
		//positions/size
		x = x + vx * t + 1/2 * ax * std::pow(t,2);
		y = y + vy * t + 1/2 * ay * std::pow(t,2);
		box_width = box_width + vw * t + 1/2 * aw * std::pow(t,2);
		box_height = box_height + vh * t + 1/2 * ah * std::pow(t,2);

		cv::Mat img(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));

		//Draw rectangle from movement
		cv::Point pt1(int(x)-int(box_width/2), int(y)-int(box_height/2));
		cv::Point pt2(int(x)+int(box_width/2), int(y)+int(box_height/2));

		cv::rectangle(img, pt1, pt2, cv::Scalar(255, 0, 0));

		float z[4*2] = {x, y, box_width, box_height, x, y, box_width, box_height};

		kalman_accel( (ap_uint<32> *)A,
				(ap_uint<32> *)Q,
				(ap_uint<32> *)Dq,
				(ap_uint<32> *)H,
				(ap_uint<32> *)X,
				(ap_uint<32> *)Up,
				(ap_uint<32> *)Dp,
				(ap_uint<32> *)R,
				(ap_uint<32> *)z,
				(ap_uint<32> *)X,
				(ap_uint<32> *)Up,
				(ap_uint<32> *)Dp,
				is_active,
				2,
					  predict_flag);

		/*for (int i = 0; i < 8; i++){
			if (X[i] != X[i+8])
			std::cout << "error: " << X[i] << " " << X[i+8] << std::endl;
		}*/

		cv::Point pt3(int(X[0])-int(X[2]/2), int(X[1])-int(X[3]/2));
		cv::Point pt4(int(X[0])+int(X[2]/2), int(X[1])+int(X[3]/2));

		cv::rectangle(img, pt3, pt4, cv::Scalar(0, 0, 255));

		cv::Point pt3_(int(X[0+8])-int(X[2+8]/2), int(X[1+8])-int(X[3+8]/2));
		cv::Point pt4_(int(X[0+8])+int(X[2+8]/2), int(X[1+8])+int(X[3+8]/2));

		cv::rectangle(img, pt3_, pt4_, cv::Scalar(0, 0, 255));

		kalman_accel( (ap_uint<32> *)A,
				(ap_uint<32> *)Q,
				(ap_uint<32> *)Dq,
				(ap_uint<32> *)H,
				(ap_uint<32> *)X,
				(ap_uint<32> *)Up,
				(ap_uint<32> *)Dp,
				(ap_uint<32> *)R,
				(ap_uint<32> *)z,
				(ap_uint<32> *)X,
				(ap_uint<32> *)Up,
				(ap_uint<32> *)Dp,
				is_active,
				2,
					  update_flag);

		cv::Point pt5(int(X[0])-int(X[2]/2), int(X[1])-int(X[3]/2));
		cv::Point pt6(int(X[0])+int(X[2]/2), int(X[1])+int(X[3]/2));

		cv::rectangle(img, pt5, pt6, cv::Scalar(255, 255, 255),2);

		cv::Point pt5_(int(X[0+8])-int(X[2+8]/2), int(X[1+8])-int(X[3+8]/2));
		cv::Point pt6_(int(X[0+8])+int(X[2+8]/2), int(X[1+8])+int(X[3+8]/2));

		cv::rectangle(img, pt5_, pt6_, cv::Scalar(255, 255, 255),2);

		if (x < 0 || x > 600 || y < 0 || y > 600){
			return 0;
		}

		std::string imname = "/home/timo/HW_Pre_Post/Tracking/Kalman/images/" + std::to_string(counter) + ".jpg";

		counter ++;

		cv::imwrite(imname, img);

	}



    return 0;

}
