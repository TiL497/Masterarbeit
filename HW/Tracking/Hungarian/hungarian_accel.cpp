#include <algorithm>
#include <limits>
#include <cmath>
#include "hungarian_accel.hpp"

#define MAX_FILTER 150

void hungarian_accel(float* detections, float *kalman_filters,  int* assignments, float *ious_, int num_detections, int num_filters) {
#pragma HLS INTERFACE m_axi     port=detections        offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=kalman_filters    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=assignments       offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=ious_             offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=num_detections
#pragma HLS INTERFACE s_axilite port=num_filters
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

	int num_jobs = std::max(num_detections, num_filters);
	float cost_matrix[MAX_FILTER*MAX_FILTER] = {0};

	for (int i = 0; i < num_detections; i++){
		for (int n = 0; n < num_filters; n++){
			float iou = calcIOU(detections[i*4], detections[i*4+2], detections[i*4+1], detections[i*4+3], kalman_filters[n*4], kalman_filters[n*4+2], kalman_filters[n*4+1], kalman_filters[n*4+3]);
			ious_[i*num_jobs+n] = iou;
			cost_matrix[i*num_jobs+n] = 1 - iou;
		}
	}

    // subtract the row minimum from each row
    for (int i = 0; i < num_jobs; i++) {
#pragma HLS LOOP_TRIPCOUNT max=150
    	float min_row = cost_matrix[i*num_jobs];
    	for (int j = 1; j < num_jobs; j++){
    		if (cost_matrix[i*num_jobs+j] < min_row){
    			min_row = cost_matrix[i*num_jobs+j];
    		}
    	}
    	for (int j = 0; j < num_jobs; j++){
    		cost_matrix[i*num_jobs+j] -= min_row;
    	}
    }

    // subtract the column minimum from each column
    for (int i = 0; i < num_jobs; i++) {
#pragma HLS LOOP_TRIPCOUNT max=150
        float min_col = cost_matrix[i];
        for (int j = 1; j < num_jobs; j++){
        	if (cost_matrix[j*num_jobs+i] < min_col){
        		min_col = cost_matrix[j*num_jobs+i];
        	}
        }
        for (int j = 0; j < num_jobs; j++){
        	cost_matrix[j*num_jobs+i] -= min_col;
        }
    }

    bool row_covered[150] = {false};
    bool col_covered[150] = {false};
    int num_assigned = 0;

    // assign detections to kalman boxes
    while (num_assigned < num_jobs) {
        for (int i = 0; i < num_jobs; i++) {
#pragma HLS LOOP_TRIPCOUNT max=150
            if (row_covered[i]) {
                continue;
            }
            for (int j = 0; j < num_jobs; j++) {
#pragma HLS LOOP_TRIPCOUNT max=150
                if (col_covered[j]) {
                    continue;
                }
                if (cost_matrix[i * num_jobs + j] == 0.0f) {
                    assignments[j] = i;
                    row_covered[i] = true;
                    col_covered[j] = true;
                    num_assigned++;
                    break;
                }
            }
        }

        float min_uncovered = std::numeric_limits<float>::max();
        for (int i = 0; i < num_jobs; i++) {
#pragma HLS LOOP_TRIPCOUNT max=150
            if (row_covered[i]) {
                continue;
            }
            for (int j = 0; j < num_jobs; j++) {
#pragma HLS LOOP_TRIPCOUNT max=150
                if (col_covered[j]) {
                    continue;
                }
                min_uncovered = std::min(min_uncovered, cost_matrix[i * num_jobs + j]);
            }
        }

        // Add the minimum uncovered value to the covered rows
        for (int i = 0; i < num_jobs; i++) {
#pragma HLS LOOP_TRIPCOUNT max=150
            if (row_covered[i]) {
                for (int j = 0; j < num_jobs; j++) {
#pragma HLS LOOP_TRIPCOUNT max=150
                    cost_matrix[i * num_jobs + j] += min_uncovered;
                }
            }
        }

        // Subtract the minimum uncovered value from the uncovered columns
        for (int j = 0; j < num_jobs; j++) {
#pragma HLS LOOP_TRIPCOUNT max=150
            if (!col_covered[j]) {
                for (int i = 0; i < num_jobs; i++) {
#pragma HLS LOOP_TRIPCOUNT max=150
                    cost_matrix[i * num_jobs + j] -= min_uncovered;
                }
            }
        }
    }
}

float calcIOU(float c_x1, float c_x2, float c_y1, float c_y2, float p_x1, float p_x2, float p_y1, float p_y2){
	//calculate overlapping points
	float xmin = std::fmax(c_x1, p_x1);
	float xmax = std::fmin(c_x2, p_x2);
	float ymin = std::fmax(c_y1, p_y1);
	float ymax = std::fmin(c_y2, p_y2);
	//compute the area of intersection rectangle
	float intersection_area = std::fmax(0, xmax - xmin + 1) * std::fmax(0, ymax - ymin + 1);
	//compute the area of both the prediction and ground-truth rectangles
	float boxAArea = (c_x2 - c_x1 + 1) * (c_y2 - c_y1 + 1);
	float boxBArea = (p_x2 - p_x1 + 1) * (p_y2 - p_y1 + 1);
	//compute the intersection over union by taking the intersection
	//area and dividing it by the sum of prediction + ground-truth
	//areas - the intersection area
	float iou = intersection_area / float(boxAArea + boxBArea - intersection_area);
	return iou;
}
