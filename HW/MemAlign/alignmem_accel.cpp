#include <cstdint>
#include "ap_int.h"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_params.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_common.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_utility.hpp"

void alignmem_accel(ap_uint<128> *in, ap_uint<128> *out){
#pragma HLS INTERFACE m_axi     port=in  	 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=out 	 offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return

	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC4, _XFCVDEPTH_DEFAULT> img_input(416, 416);
	xf::cv::Mat<XF_8UC3, 416, 416, XF_NPPC4, _XFCVDEPTH_DEFAULT> img_out(416, 416);

#pragma HLS DATAFLOW

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
