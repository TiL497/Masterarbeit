#include <cstdint>
#include "hls_stream.h"

#define N_CLASSES 1
#define MAX_DET 50
#define MAX_PROPOSALS 500
#define IMAGE_SIZE 416
#define SIZE_L 13
#define SIZE_M 26
#define SIZE_S 52
#define GRID_SCALE_L IMAGE_SIZE/SIZE_L
#define GRID_SCALE_M IMAGE_SIZE/SIZE_M
#define GRID_SCALE_S IMAGE_SIZE/SIZE_S

void postprocess_accel(int8_t *results_L, int8_t *results_M, int8_t *results_S, int *items_out, float *box_out, uint16_t bboxes_L[6], uint16_t bboxes_M[6], uint16_t bboxes_S[6], float data_scale_L, float data_scale_M, float data_scale_S, float obj_threshold, float nms_threshold);
void sortObjScores(hls::stream<float> &obj_to_sort, hls::stream<float> &x1_to_sort, hls::stream<float> &y1_to_sort, hls::stream<float> &x2_to_sort, hls::stream<float> &y2_to_sort, hls::stream<bool> &sort_end, float arr_obj_scores[MAX_PROPOSALS], float arr_x1[MAX_PROPOSALS], float arr_y1[MAX_PROPOSALS], float arr_x2[MAX_PROPOSALS], float arr_y2[MAX_PROPOSALS]);
float calcIOU(float c_x1, float c_x2, float c_y1, float c_y2, float p_x1, float p_x2, float p_y1, float p_y2);
