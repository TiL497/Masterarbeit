#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_params.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_utility.hpp"

// Set the pixel depth:
#define TYPE XF_32FC1
#define PTR_WIDTH 32

// Set the optimization type:
#define NPC1 XF_NPPC1

#define KF_N 8
#define KF_M 4
#define KF_C 0
#define KF_MTU 1
#define KF_MMU 1
#define XF_USE_URAM 0
#define KF_EKF 0
#define XF_CV_DEPTH_A 3
#define XF_CV_DEPTH_B 3
#define XF_CV_DEPTH_UQ 3
#define XF_CV_DEPTH_DQ 3
#define XF_CV_DEPTH_H 3
#define XF_CV_DEPTH_X0 3
#define XF_CV_DEPTH_U0 3
#define XF_CV_DEPTH_D0 3
#define XF_CV_DEPTH_R 3
#define XF_CV_DEPTH_U 3
#define XF_CV_DEPTH_Y 3
#define XF_CV_DEPTH_XOUT 3
#define XF_CV_DEPTH_UOUT 3
#define XF_CV_DEPTH_DOUT 3

void kalman_accel(ap_uint<32> * in_A,
				  ap_uint<32>* in_Uq,
				  ap_uint<32>* in_Dq,
				  ap_uint<32>* in_H,
				  ap_uint<32>* in_X0,
				  ap_uint<32>* in_U0,
				  ap_uint<32>* in_D0,
				  ap_uint<32>* in_R,
				  ap_uint<32>* in_y,
				  ap_uint<32>* out_X,
				  ap_uint<32>* out_U,
				  ap_uint<32>* out_D,
				  bool* is_active,
				  int num_filter,
				  unsigned char control_flag);
