#include "ap_int.h"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_common.hpp"
#include "/home/timo/Vitis_Libraries/vision/L1/include/common/xf_utility.hpp"

#define PTR_WIDTH_IN 256
#define PTR_WIDTH_OUT 256
#define rows_in 720
#define cols_in 1280
#define rows_out 416
#define cols_out 416


void preprocess_accel(ap_uint<PTR_WIDTH_IN> *img_in, ap_uint<PTR_WIDTH_OUT> *img_out, unsigned char scale);
template <int TYPE,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH>
void prescale_accel(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH>& data_in, xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH>& data_out, unsigned char scale);
template <int TYPE,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH>
void tracks_accel(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH>& data, float *tracks, ap_uint<24> *color, int num_tracks);
void alignmem_accel(ap_uint<PTR_WIDTH_OUT> *in, ap_uint<PTR_WIDTH_OUT> *out);
