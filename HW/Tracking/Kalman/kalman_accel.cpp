#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_params.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_common.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_utility.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/video/xf_kalmanfilter.hpp"
#include "kalman_accel.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include <cmath>
#include "iostream"

static constexpr int __XF_DEPTH_NN = KF_N * KF_N;
static constexpr int __XF_DEPTH_NC = KF_N * KF_C;
static constexpr int __XF_DEPTH_MN = KF_M * KF_N;
static constexpr int __XF_DEPTH_N1 = KF_N;
static constexpr int __XF_DEPTH_M1 = KF_M;
static constexpr int __XF_DEPTH_C1 = KF_C;

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
				  unsigned char control_flag){

#pragma HLS INTERFACE m_axi      port=in_A     	    offset=slave  bundle=gmem0
#pragma HLS INTERFACE m_axi      port=in_Uq     	offset=slave  bundle=gmem1
#pragma HLS INTERFACE m_axi      port=in_Dq     	offset=slave  bundle=gmem2
#pragma HLS INTERFACE m_axi      port=in_H     	    offset=slave  bundle=gmem3
#pragma HLS INTERFACE m_axi      port=in_X0     	offset=slave  bundle=gmem4
#pragma HLS INTERFACE m_axi      port=in_U0     	offset=slave  bundle=gmem5
#pragma HLS INTERFACE m_axi      port=in_D0      	offset=slave  bundle=gmem6
#pragma HLS INTERFACE m_axi      port=in_R      	offset=slave  bundle=gmem7
#pragma HLS INTERFACE m_axi      port=in_y      	offset=slave  bundle=gmem8
#pragma HLS INTERFACE m_axi      port=out_X      	offset=slave  bundle=gmem9
#pragma HLS INTERFACE m_axi      port=out_U      	offset=slave  bundle=gmem10
#pragma HLS INTERFACE m_axi      port=out_D        	offset=slave  bundle=gmem11
#pragma HLS INTERFACE s_axilite  port=control_flag
#pragma HLS INTERFACE s_axilite  port=return

    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_A> A_mat(KF_N, KF_N);
    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_UQ> Uq_mat(KF_N, KF_N);
    xf::cv::Mat<TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_DQ> Dq_mat(KF_N, 1);
    xf::cv::Mat<TYPE, KF_M, KF_N, NPC1, XF_CV_DEPTH_H> H_mat(KF_M, KF_N);
    xf::cv::Mat<TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_X0> x_k1_mat(KF_N, 1);
    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_U0> Up_mat(KF_N, KF_N);
    xf::cv::Mat<TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_D0> Dp_mat(KF_N, 1);
    xf::cv::Mat<TYPE, KF_M, 1, NPC1, XF_CV_DEPTH_R> R_mat(KF_M, 1);
    xf::cv::Mat<TYPE, KF_M, 1, NPC1, XF_CV_DEPTH_Y> z_mat(KF_M, 1);

    xf::cv::Mat<TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_XOUT> x_k_mat(KF_N, 1);
    xf::cv::Mat<TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_UOUT> Up_out_mat(KF_N, KF_N);
    xf::cv::Mat<TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_DOUT> Dp_out_mat(KF_N, 1);

#pragma HLS DATAFLOW

	xf::cv::Array2xfMat<32, TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_A>(in_A, A_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_UQ>(in_Uq, Uq_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_DQ>(in_Dq, Dq_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_M, KF_N, NPC1, XF_CV_DEPTH_H>(in_H, H_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_M, 1, NPC1, XF_CV_DEPTH_R>(in_R, R_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_X0>(in_X0, x_k1_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_U0>(in_U0, Up_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_D0>(in_D0, Dp_mat);
	xf::cv::Array2xfMat<32, TYPE, KF_M, 1, NPC1, XF_CV_DEPTH_Y>(in_y, z_mat);

	// Kalman Filter
	xf::cv::KalmanFilter<KF_N, KF_M, KF_C, KF_MTU, KF_MMU, XF_USE_URAM, 0, TYPE, NPC1, XF_CV_DEPTH_A, XF_CV_DEPTH_B,
						 XF_CV_DEPTH_UQ, XF_CV_DEPTH_DQ, XF_CV_DEPTH_H, XF_CV_DEPTH_X0, XF_CV_DEPTH_U0, XF_CV_DEPTH_D0,
						 XF_CV_DEPTH_R, XF_CV_DEPTH_U, XF_CV_DEPTH_Y, XF_CV_DEPTH_XOUT, XF_CV_DEPTH_UOUT,
						 XF_CV_DEPTH_DOUT>(A_mat, Uq_mat, Dq_mat, H_mat, x_k1_mat, Up_mat, Dp_mat, R_mat,
										   z_mat, x_k_mat, Up_out_mat, Dp_out_mat, control_flag);

	//Output the data
	xf::cv::xfMat2Array<32, TYPE, KF_N, KF_N, NPC1, XF_CV_DEPTH_UOUT>(Up_out_mat, out_U);
	xf::cv::xfMat2Array<32, TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_DOUT>(Dp_out_mat, out_D);
	xf::cv::xfMat2Array<32, TYPE, KF_N, 1, NPC1, XF_CV_DEPTH_XOUT>(x_k_mat, out_X);


}
