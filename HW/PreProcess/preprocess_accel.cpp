#include "ap_int.h"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_params.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_common.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_utility.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/imgproc/xf_resize.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/imgproc/xf_cvt_color.hpp"
#include "preprocess_accel.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/dnn/xf_insertBorder.hpp"
#include <cstdint>

void preprocess_accel(ap_uint<PTR_WIDTH_IN> *img_in, ap_uint<PTR_WIDTH_OUT> *img_out, unsigned char scale){
#pragma HLS INTERFACE m_axi     port=img_in  	 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=img_out 	 offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=scale
#pragma HLS INTERFACE s_axilite port=return

	// Compute Resize output image size for Letterbox
	float scale_height = (float)rows_out/(float)rows_in;
	float scale_width = (float)cols_out/(float)cols_in;
	int rows_out_resize, cols_out_resize;
	if(scale_width<scale_height){
		cols_out_resize = cols_out;
		rows_out_resize = (int)((float)(rows_in*cols_out)/(float)cols_in);
	}
	else{
		cols_out_resize = (int)((float)(cols_in*rows_out)/(float)rows_in);
		rows_out_resize = rows_out;
	}

	//generate cv::Mat
	xf::cv::Mat<XF_8UC3, 720, 1280, XF_NPPC8, _XFCVDEPTH_DEFAULT> img_input(rows_in, cols_in);
	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT> out_mat_resize(rows_out_resize, cols_out_resize);
	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT> out_mat_bgr2rgb(rows_out, cols_out);
	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT> resized_mat(rows_out, cols_out);
	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT> img_output(rows_out, cols_out);

#pragma HLS DATAFLOW

	//do resize
	xf::cv::Array2xfMat<PTR_WIDTH_IN,XF_8UC3, 720, 1280, XF_NPPC8, _XFCVDEPTH_DEFAULT>  ((ap_uint<PTR_WIDTH_IN> *)img_in, img_input);
	xf::cv::resize<XF_INTERPOLATION_BILINEAR,XF_8UC3, 720, 1280, 416, 416,XF_NPPC8, _XFCVDEPTH_DEFAULT, _XFCVDEPTH_DEFAULT, 5> (img_input, out_mat_resize);
	xf::cv::insertBorder<XF_8UC3, 416, 416, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT, _XFCVDEPTH_DEFAULT>(out_mat_resize, out_mat_bgr2rgb, 0);
	xf::cv::bgr2rgb<XF_8UC3,XF_8UC3, 416, 416,XF_NPPC8, _XFCVDEPTH_DEFAULT, _XFCVDEPTH_DEFAULT>(out_mat_bgr2rgb,resized_mat);
	prescale_accel<XF_8UC3, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT> (resized_mat, img_output, scale);
	xf::cv::xfMat2Array<PTR_WIDTH_OUT, XF_8UC3, 416, 416, XF_NPPC8, _XFCVDEPTH_DEFAULT>(img_output, img_out);

}

template <int TYPE,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH>
void prescale_accel(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH>& data_in, xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH>& data_out, unsigned char scale){

	enum { DEPTH = TYPE, PLANES = XF_CHANNELS(TYPE, NPC) };

	//change for different sizes
	unsigned int data_size = ROWS*COLS/NPC;
	for(int i=0; i < data_size; i++){
		//pixel level
        XF_TNAME(DEPTH, NPC) out_pix;
		XF_TNAME(DEPTH, NPC) in_pix;
        ap_uint<NPC * PLANES * 8> out_plane_tmp;
		//read data from Mat
		in_pix = data_in.read(i);
		//pixel level
		for(int pixel=0, pixel_bit = 0; pixel < NPC; pixel++, pixel_bit += (PLANES*8)){
			ap_uint<PLANES * 8> in_plane_tmp;
			//read all three channels of pixel
			in_plane_tmp = in_pix.range(pixel_bit + (PLANES*8) - 1, pixel_bit);
			//channel level
			for(int channel=0, channel_bit = 0; channel < PLANES; channel++, channel_bit += 8){
				//scale pixel channel-wise
				float temp = float(in_plane_tmp.range(channel_bit + 7, channel_bit))/255.0;
				temp = temp * (float)scale;
				out_plane_tmp.range(pixel_bit + channel_bit + 7, pixel_bit + channel_bit) = (unsigned char)temp;
				//out_plane_tmp.range(pixel_bit + channel_bit + 7, pixel_bit + channel_bit) = (unsigned char)(float(float(in_plane_tmp.range(channel_bit + 7, channel_bit))/255.0)*scale);
			}
		}
		//write data to Mat
		out_pix = out_plane_tmp;
		data_out.write(i, out_pix);
	}
	return;
}

void alignmem_accel(ap_uint<128> *in, ap_uint<128> *out){

	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC4, _XFCVDEPTH_DEFAULT> img_input(416, 416);
	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC4, _XFCVDEPTH_DEFAULT> img_out(416, 416);

	xf::cv::Array2xfMat<128,XF_8UC3, 416, 416, XF_NPPC4, _XFCVDEPTH_DEFAULT>  (in, img_input);
	enum { DEPTH = XF_8UC3};

	//change for different sizes
	unsigned int data_size = 416*416/XF_NPPC4;
	for(int i=0; i < data_size; i++){
		//pixel level
        XF_TNAME(DEPTH, XF_NPPC4) out_pix;
		XF_TNAME(DEPTH, XF_NPPC4) in_pix;
		//read data from Mat
		in_pix = img_input.read(i);
		//write data to Mat
		img_out.write(i, in_pix);
	}
	xf::cv::xfMat2Array<128, XF_8UC3, 416, 416, XF_NPPC4, _XFCVDEPTH_DEFAULT>(img_out, out);

}
