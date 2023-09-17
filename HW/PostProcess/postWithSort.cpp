#include "postWithSort.hpp"
#include <cstdint>
#include "hls_stream.h"
#include <cmath>
#include <ap_int.h>
#include <iostream>
#include "hls_math.h"

void postprocess_accel(int8_t *results_L, int8_t *results_M, int8_t *results_S, int *items_out, float *box_out, uint16_t bboxes_L[6], uint16_t bboxes_M[6], uint16_t bboxes_S[6], float data_scale_L, float data_scale_M, float data_scale_S, float obj_threshold, float nms_threshold){
#pragma HLS INTERFACE m_axi     port=results_L   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=results_M   offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=results_S   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=items_out   offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=box_out     offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=bboxes_L
#pragma HLS INTERFACE s_axilite port=bboxes_M
#pragma HLS INTERFACE s_axilite port=bboxes_S
#pragma HLS INTERFACE s_axilite port=data_scale_L
#pragma HLS INTERFACE s_axilite port=data_scale_M
#pragma HLS INTERFACE s_axilite port=data_scale_S
#pragma HLS INTERFACE s_axilite port=obj_threshold
#pragma HLS INTERFACE s_axilite port=return

	int feature_size = 5+N_CLASSES;
	int num_proposals = 0;
	int current_idx = 0;
	float arr_obj_scores [MAX_PROPOSALS];
	float arr_x1 [MAX_PROPOSALS];
	float arr_x2 [MAX_PROPOSALS];
	float arr_y1 [MAX_PROPOSALS];
	float arr_y2 [MAX_PROPOSALS];

	hls::stream<float, MAX_PROPOSALS> obj_to_sort;
	hls::stream<float, MAX_PROPOSALS> x1_to_sort;
	hls::stream<float, MAX_PROPOSALS> y1_to_sort;
	hls::stream<float, MAX_PROPOSALS> x2_to_sort;
	hls::stream<float, MAX_PROPOSALS> y2_to_sort;
	hls::stream<bool, MAX_PROPOSALS> end_stream;

	for (int r = 0; r < SIZE_S; r++){
		for (int c = 0; c < SIZE_S; c++){
			for (int anchor = 0; anchor < 3; anchor++){
				float obj_score = 1/(1+std::exp(-1*float(results_S[(r*SIZE_S*3+c*3+anchor)*feature_size+4])*data_scale_S));
				if(obj_score > obj_threshold && num_proposals < MAX_PROPOSALS){
					//y center
					float y_center = (1/(1+std::exp(-1*float(results_S[(r*SIZE_S*3+c*3+anchor)*feature_size+1])*data_scale_S))*2.0 - 0.5 + r)*GRID_SCALE_S;
					//x_center
					float x_center = (1/(1+std::exp(-1*float(results_S[(r*SIZE_S*3+c*3+anchor)*feature_size])*data_scale_S))*2.0 - 0.5 + c)*GRID_SCALE_S;
					//width
					float width = std::pow((1/(1+std::exp(-1*float(results_S[(r*SIZE_S*3+c*3+anchor)*feature_size+2])*data_scale_S)))*2,2)*bboxes_S[2*anchor+0];
					//height
					float height = std::pow((1/(1+std::exp(-1*float(results_S[(r*SIZE_S*3+c*3+anchor)*feature_size+3])*data_scale_S)))*2,2)*bboxes_S[2*anchor+1];
					//x1,y1,x2,y2 of bbox
					end_stream.write(1);
					obj_to_sort.write(obj_score);
					x1_to_sort.write((x_center - width/2));
					y1_to_sort.write((y_center - height/2));
					x2_to_sort.write((x_center + width/2));
					y2_to_sort.write((y_center + height/2));
					num_proposals ++;
				}
			}
		}
	}

	for (int r = 0; r < SIZE_M; r++){
		for (int c = 0; c < SIZE_M; c++){
			for (int anchor = 0; anchor < 3; anchor++){
				float obj_score = 1/(1+std::exp(-1*float(results_M[(r*SIZE_M*3+c*3+anchor)*feature_size+4])*data_scale_M));
				if(obj_score > obj_threshold && num_proposals < MAX_PROPOSALS){
					//y_center
					float y_center = (1/(1+std::exp(-1*float(results_M[(r*SIZE_M*3+c*3+anchor)*feature_size+1])*data_scale_M))*2.0 - 0.5 + r)*GRID_SCALE_M;
					//x_center
					float x_center = (1/(1+std::exp(-1*float(results_M[(r*SIZE_M*3+c*3+anchor)*feature_size])*data_scale_M))*2.0 - 0.5 + c)*GRID_SCALE_M;
					//width
					float width = std::pow((1/(1+std::exp(-1*float(results_M[(r*SIZE_M*3+c*3+anchor)*feature_size+2])*data_scale_M)))*2,2)*bboxes_M[2*anchor+0];
					//height
					float height = std::pow((1/(1+std::exp(-1*float(results_M[(r*SIZE_M*3+c*3+anchor)*feature_size+3])*data_scale_M)))*2,2)*bboxes_M[2*anchor+1];
					//x1,y1,x2,y2 of bbox
					end_stream.write(1);
					obj_to_sort.write(obj_score);
					x1_to_sort.write((x_center - width/2));
					y1_to_sort.write((y_center - height/2));
					x2_to_sort.write((x_center + width/2));
					y2_to_sort.write((y_center + height/2));
					num_proposals ++;
				}
			}
		}
	}

	for (int r = 0; r < SIZE_L; r++){
		for (int c = 0; c < SIZE_L; c++){
			for (int anchor = 0; anchor < 3; anchor++){
				float obj_score = 1/(1+std::exp(-1*float(results_L[(r*SIZE_L*3+c*3+anchor)*feature_size+4])*data_scale_L));
				if(obj_score > obj_threshold && num_proposals < MAX_PROPOSALS){
					//y_center
					float y_center = (1/(1+std::exp(-1*float(results_L[(r*SIZE_L*3+c*3+anchor)*feature_size+1])*data_scale_L))*2.0 - 0.5 + r)*GRID_SCALE_L;
					//x_center
					float x_center = (1/(1+std::exp(-1*float(results_L[(r*SIZE_L*3+c*3+anchor)*feature_size])*data_scale_L))*2.0 - 0.5 + c)*GRID_SCALE_L;
					//width
					float width = std::pow((1/(1+std::exp(-1*float(results_L[(r*SIZE_L*3+c*3+anchor)*feature_size+2])*data_scale_L)))*2,2)*bboxes_L[2*anchor+0];
					//height
					float height = std::pow((1/(1+std::exp(-1*float(results_L[(r*SIZE_L*3+c*3+anchor)*feature_size+3])*data_scale_L)))*2,2)*bboxes_L[2*anchor+1];
					//x1,y1,x2,y2 of bbox
					end_stream.write(1);
					obj_to_sort.write(obj_score);
					x1_to_sort.write((x_center - width/2));
					y1_to_sort.write((y_center - height/2));
					x2_to_sort.write((x_center + width/2));
					y2_to_sort.write((y_center + height/2));
					num_proposals ++;
				}
			}
		}
	}

	end_stream.write(0);

	sortObjScores(obj_to_sort, x1_to_sort, y1_to_sort, x2_to_sort, y2_to_sort, end_stream, arr_obj_scores, arr_x1, arr_y1, arr_x2, arr_y2);

	for (int i = 0; i < num_proposals; i++){
		if (arr_obj_scores[i] == 0){
			continue;
		}
		box_out[current_idx*8+0] = arr_x1[i];
		box_out[current_idx*8+1] = arr_y1[i];
		box_out[current_idx*8+2] = arr_x2[i];
		box_out[current_idx*8+3] = arr_y2[i];
		box_out[current_idx*8+4] = 0;
		box_out[current_idx*8+5] = 0;
		box_out[current_idx*8+6] = arr_obj_scores[i];
		box_out[current_idx*8+7] = 0;
		arr_obj_scores[i] = 0;
		for (int j = i+1; j < num_proposals; j++){
			if (arr_obj_scores[j] == 0){
				continue;
			}
			if (calcIOU(arr_x1[i], arr_x2[i], arr_y1[i], arr_y2[i], arr_x1[j], arr_x2[j], arr_y1[j], arr_y2[j]) > nms_threshold){
				arr_obj_scores[j] = 0;
			}
		}
		current_idx ++;
		if (current_idx >= MAX_DET){
			break;
		}
	}
	*items_out = current_idx;
}

void sortObjScores(hls::stream<float> &obj_to_sort, hls::stream<float> &x1_to_sort, hls::stream<float> &y1_to_sort, hls::stream<float> &x2_to_sort, hls::stream<float> &y2_to_sort, hls::stream<bool> &sort_end, float arr_obj_scores[MAX_PROPOSALS], float arr_x1[MAX_PROPOSALS], float arr_y1[MAX_PROPOSALS], float arr_x2[MAX_PROPOSALS], float arr_y2[MAX_PROPOSALS]){

	int items = 0;
	bool flag_end_stream;
	flag_end_stream = sort_end.read();
	while (flag_end_stream){
		//sort
		float score, x1, y1, x2 ,y2;
		score = obj_to_sort.read();
		x1 = x1_to_sort.read();
		y1 = y1_to_sort.read();
		x2 = x2_to_sort.read();
		y2 = y2_to_sort.read();
		if (items == 0){
			arr_obj_scores[items] = score;
			arr_x1[items] = x1;
			arr_y1[items] = y1;
			arr_x2[items] = x2;
			arr_y2[items] = y2;
		}
		for (int n = items; n > 0; n--){
			if (score > arr_obj_scores[n-1]){
				float tmp_score = arr_obj_scores[n-1];
				arr_obj_scores[n-1] = score;
				arr_obj_scores[n] = tmp_score;
				float tmp_x1 = arr_x1[n-1];
				arr_x1[n-1] = x1;
				arr_x1[n] = tmp_x1;
				float tmp_y1 = arr_y1[n-1];
				arr_y1[n-1] = y1;
				arr_y1[n] = tmp_y1;
				float tmp_x2 = arr_x2[n-1];
				arr_x2[n-1] = x2;
				arr_x2[n] = tmp_x2;
				float tmp_y2 = arr_y2[n-1];
				arr_y2[n-1] = y2;
				arr_y2[n] = tmp_y2;
			} else {
				arr_obj_scores[n] = score;
				arr_x1[n] = x1;
				arr_y1[n] = y1;
				arr_x2[n] = x2;
				arr_y2[n] = y2;
				break;
			}
		}
		items ++;
		flag_end_stream = sort_end.read();
	}
}

float calcIOU(float c_x1, float c_x2, float c_y1, float c_y2, float p_x1, float p_x2, float p_y1, float p_y2){
	//calculate overlapping points
	float xA = std::fmax(c_x1, p_x1);
	float yA = std::fmax(c_y1, p_y1);
	float xB = std::fmin(c_x2, p_x2);
	float yB = std::fmin(c_y2, p_y2);
	//compute the area of intersection rectangle
	float intersection_area = std::fmax(0, xB - xA + 1) * std::fmax(0, yB - yA + 1);
	//compute the area of both the prediction and ground-truth rectangles
	float boxAArea = (c_x2 - c_x1 + 1) * (c_y2 - c_y1 + 1);
	float boxBArea = (p_x2 - p_x1 + 1) * (p_y2 - p_y1 + 1);
	//compute the intersection over union by taking the intersection
	//area and dividing it by the sum of prediction + ground-truth
	//areas - the intersection area
	float iou = intersection_area / float(boxAArea + boxBArea - intersection_area);
	return iou;
}
