#pragma once 
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <vector>
#include <tuple>
#include <string>

//constants for inference and tracking 
#define NMS_THRESHOLD 0.3f
#define OBJ_THRESHOLD 0.2f
#define OBJ_HIGH 0.25f 
#define TRACKING_THRESHOLD 0.3f
//constants for Post-Processing
#define N_CLASSES 1
#define MAX_PROPOSALS 500
#define MAX_DET 50 
#define TENSOR_SIZE_L 13
#define TENSOR_SIZE_M 26
#define TENSOR_SIZE_S 52
#define GRID_SCALE_L 416/TENSOR_SIZE_L
#define GRID_SCALE_M 416/TENSOR_SIZE_M
#define GRID_SCALE_S 416/TENSOR_SIZE_S

// Structs for configuration parameters  
typedef struct ProgConfig{
    int grid_scale_L; 
    int grid_scale_M; 
    int grid_scale_S; 
    int max_boxes; 
    int img_in_width; 
    int img_in_height; 
    int img_out_width; 
    int img_out_height; 
    uint16_t *bboxes_L;
    uint16_t *bboxes_M;
    uint16_t *bboxes_S;
    uint8_t processing; 
    uint8_t source; 
    uint64_t data_in_phy_addr;
    uint64_t data_in_addr;
    uint64_t data_out_phy_addr[3];
    double timings_preproc[1000] = {0};
    double timings_postproc[1000] = {0};
    double timings_dpu[1000] = {0};
    double timings_kalman[1000] = {0};
    double timings_total[1000] = {0};
    int current_idx = 0;
} ProgConfig;

//Software implementation of Post-Processing
void postProcess(int8_t *results_L, int8_t *results_M, int8_t *results_S, float data_scale_L, float data_scale_M, float data_scale_S, float obj_threshold, float nms_threshold, uint16_t *bboxes_L, uint16_t *bboxes_M, uint16_t *bboxes_S, int *items_out, float *box_out);
//Software implementation of Intersection over Union 
float calcIOU(float c_x1, float c_x2, float c_y1, float c_y2, float p_x1, float p_x2, float p_y1, float p_y2);
//Runner Function that executes the program and calls various functions for HW-Kernels 
int runDPU(uint8_t source, uint8_t processing, const char *model, const char *dpu_path, const char *img_path, const char *roi_path);
//Software implementation of Pre-Processing
void preProcess(const cv::Mat &src, cv::Mat &dst, int in_width, int in_height, int out_width, int out_height, unsigned char scaling_factor);
//Function to compute timings  
void computeTimings(ProgConfig cfg);
//Software implementation of the hungarian algorithm
void hungarianAlgorithm(float* cost_matrix, int num_jobs, int* assignments);