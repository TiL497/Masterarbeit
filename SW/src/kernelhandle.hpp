#include "common.hpp"
#include "DPU.hpp"
#include "cstdint"
#include <string>
// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
//Vitis Ai includes 
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
//Object Tracking includes 
#include "kalman.hpp"

//define sizes to initialize kernel 
#define pre_in_size   1280*720*3
#define pre_out_size  416*416*3
#define post_in_size_L  13*13*3*8
#define post_in_size_M  26*26*3*8
#define post_in_size_S  52*52*3*8
#define post_out_size MAX_DET*8
#define predict_flag 1+2+8+16
#define update_flag 1+4+32+64

// Struct for Pre-Processing Handle  
typedef struct PreKernelHandle{
    xrt::kernel kernel; 
    xrt::run runner;
    xrt::run runner_align; 
    xrt::device device; 
    xrt::bo data_in;
    xrt::bo data_out;
    unsigned char *data_in_m; 
    unsigned char *data_out_m; 
    unsigned char scale; 
} PreKernelHandle;

// Struct for Post-Processing Handle  
typedef struct PostKernelHandle{
    xrt::kernel kernel; 
    xrt::run runner; 
    xrt::device device; 
    xrt::bo in_results_L;
    xrt::bo in_results_M;
    xrt::bo in_results_S;
    xrt::bo items; 
    xrt::bo data_out;
    int8_t *in_results_L_m;
    int8_t *in_results_M_m;
    int8_t *in_results_S_m;
    int *items_m;  
    float *data_out_m; 
    float scale_small;
    float scale_medium;
    float scale_large;
} PostKernelHandle;

// Struct for DPU Handle  
typedef struct DPUKernelHandle{
    std::unique_ptr<vart::RunnerExt> dpu_runner;
    std::vector<vart::TensorBuffer*> input_tensor_buffers;
    std::vector<vart::TensorBuffer*> output_tensor_buffers;
} DPUKernelHandle;

// Struct for Kalman Handle  
typedef struct KalmanKernelHandle{
    xrt::kernel kernel; 
    xrt::run runner; 
    xrt::device device; 
    xrt::bo in_A; 
    xrt::bo in_Uq; 
    xrt::bo in_Dq; 
    xrt::bo in_H; 
    xrt::bo in_x_k1; 
    xrt::bo in_Up; 
    xrt::bo in_Dp;
    xrt::bo in_R; 
    xrt::bo in_z; 
    xrt::bo out_x_k; 
    xrt::bo out_Up; 
    xrt::bo out_Dp; 
    float *in_A_m; 
    float *in_Uq_m; 
    float *in_Dq_m; 
    float *in_H_m; 
    float *in_x_k1_m; 
    float *in_Up_m; 
    float *in_Dp_m;
    float *in_R_m; 
    float *in_z_m; 
    float *out_x_k_m; 
    float *out_Up_m; 
    float *out_Dp_m; 
} KalmanKernelHandle;

// Struct for Hungarian Handle  
typedef struct HungarianHandle{
    xrt::kernel kernel; 
    xrt::run runner; 
    xrt::device device; 
    xrt::bo detections; 
    xrt::bo kalman_filters; 
    xrt::bo assignments; 
    xrt::bo ious; 
    float *detections_m; 
    float *kalman_filters_m; 
    float *ious_m; 
    int *assignments_m; 
} HungarianHandle;

//Function to initialize HW-Kernels 
void prepostKernelInit(PreKernelHandle *preproc_handle, PostKernelHandle *postproc_handle, KalmanKernelHandle *kalman_handle, HungarianHandle *hungarian_handle, const char *path);
//Function to invoke DPU and all other Kernels 
void invokeDPU(DPUKernelHandle *dpu_handle, PreKernelHandle *preproc_handle, PostKernelHandle *postproc_handle, KalmanKernelHandle *kalman_handle, HungarianHandle *hungarian_handle, cv::Mat &img, cv::Mat &img_resized, cv::Mat &roi, int8_t *results_L, int8_t *results_M, int8_t *results_S, int *items, float *box_out, ProgConfig *cfg, std::string imname, float obj_threshold, float nms_threshold, std::vector<KalmanFilter> &kalman_tracks, std::vector<Tracking> &trackings, int *tracking_id);
//Function to invoke Pre-Processing Kernel  
void invokePreKernel(PreKernelHandle *preproc_handle, unsigned char *data, uint64_t dpu_input_addr);
//Function to invoke Post-Processing Kernel  
void invokePostKernel(PostKernelHandle *postproc_handle, uint16_t *bboxes_L, uint16_t *bboxes_M, uint16_t *bboxes_S, float obj_threshold, float nms_threshold, uint64_t addr[3]); 