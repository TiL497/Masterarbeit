#include "kernelhandle.hpp"
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
#include <fstream>
#include <iostream>
#include <string>
//Object tracking includes
#include "kalman.cpp"

/**
 * @brief prepostKernelInit 
 *
 * Function to initialize HW-Kernels 
 *
 * @param *preproc_handle[inout]: Handle of Pre-Processing Kernel 
 * @param *postproc_handle[inout]: Handle of Post-Processing Kernel 
 * @param *kalman_handle[inout]: Handle of Kalman Kernel 
 * @param *hungarian_handle[inout]: Handle of Hungarian Kernel 
 * @param *path[in]: path to xclbin containing information about the hardware  
 *  
 */
void prepostKernelInit(PreKernelHandle *preproc_handle, PostKernelHandle *postproc_handle, KalmanKernelHandle *kalman_handle, HungarianHandle *hungarian_handle, const char *path){
	// Device/Card Index on system
	unsigned device_index = 0;
	
	// Acquire Device by index
	auto device = xrt::device(device_index); 
	// Load XCLBIN
	auto uuid = device.load_xclbin(path);    
	// Get Kernel/Pre-Post-Proc CU	
	auto postproc_accelerator = xrt::kernel(device, uuid, "postprocess_accel");	
	auto postproc_runner = xrt::run(postproc_accelerator);
	auto alignmem_accelerator = xrt::kernel(device, uuid, "alignmem_accel");	
	auto alignmem_runner = xrt::run(alignmem_accelerator);
	auto preproc_accelerator = xrt::kernel(device, uuid, "preprocess_accel");	
	auto preproc_runner = xrt::run(preproc_accelerator);
	auto kalman_accelerator = xrt::kernel(device, uuid, "kalman_accel");	
	auto kalman_runner = xrt::run(kalman_accelerator);
	auto hungarian_accelerator = xrt::kernel(device, uuid, "hungarian_accel");	
	auto hungarian_runner = xrt::run(hungarian_accelerator);

	//define sizes for Kernel 
	const int pre_in_data_size = pre_in_size*sizeof(unsigned char);
	const int pre_out_data_size = pre_out_size*sizeof(unsigned char);
	const int post_in_data_size_L = post_in_size_L*sizeof(int8_t);
	const int post_in_data_size_M = post_in_size_M*sizeof(int8_t);
	const int post_in_data_size_S = post_in_size_S*sizeof(int8_t);
	const int items_size = sizeof(int);
	const int bboxes_size = 6*sizeof(uint16_t);
	const int post_out_data_size = post_out_size*sizeof(float);
	const int indexes_size = 500*sizeof(int); 
	
	//create Buffer for communication with Pre-Proc Kernel 
	auto img_preproc_in = xrt::bo(device, pre_in_data_size, preproc_accelerator.group_id(0));
	auto img_preproc_in_m = img_preproc_in.map<unsigned char *>();
	if (img_preproc_in_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");   
    }   

	auto img_preproc_out = xrt::bo(device, pre_out_data_size, alignmem_accelerator.group_id(0));
	auto img_preproc_out_m  = img_preproc_out.map<unsigned char *>();
	if (img_preproc_out_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	//create Buffer for communication with Post-Proc Kernel 
	auto items = xrt::bo(device, items_size, postproc_accelerator.group_id(3));
	auto items_m  = items.map<int *>();
	if (items_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto out_postproc = xrt::bo(device, post_out_data_size, postproc_accelerator.group_id(4));
	auto out_postproc_m  = out_postproc.map<float *>();
	if (out_postproc_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	//create Buffer for communication with Kalman Kernel 
	auto in_A = xrt::bo(device, DIM*DIM*sizeof(float), kalman_accelerator.group_id(0));
	auto in_A_m = in_A.map<float *>();
	if (in_A_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");   
    }   

	auto in_Uq = xrt::bo(device, DIM*DIM*sizeof(float), kalman_accelerator.group_id(1));
	auto in_Uq_m  = in_Uq.map<float *>();
	if (in_Uq_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_Dq = xrt::bo(device, DIM*sizeof(float), kalman_accelerator.group_id(2));
	auto in_Dq_m  = in_Dq.map<float *>();
	if (in_Dq_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_H = xrt::bo(device, 4*DIM*sizeof(float), kalman_accelerator.group_id(3));
	auto in_H_m  = in_H.map<float *>();
	if (in_H_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_x_k1 = xrt::bo(device, DIM*sizeof(float), kalman_accelerator.group_id(4));
	auto in_x_k1_m  = in_x_k1.map<float *>();
	if (in_x_k1_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_Up = xrt::bo(device, DIM*DIM*sizeof(float), kalman_accelerator.group_id(5));
	auto in_Up_m  = in_Up.map<float *>();
	if (in_Up_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_Dp = xrt::bo(device, DIM*sizeof(float), kalman_accelerator.group_id(6));
	auto in_Dp_m  = in_Dp.map<float *>();
	if (in_Dp_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_R = xrt::bo(device, DIM*sizeof(float)/2, kalman_accelerator.group_id(7));
	auto in_R_m  = in_R.map<float *>();
	if (in_R_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto in_z = xrt::bo(device, DIM*sizeof(float)/2, kalman_accelerator.group_id(8));
	auto in_z_m  = in_z.map<float *>();
	if (in_z_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto out_x_k = xrt::bo(device, DIM*sizeof(float), kalman_accelerator.group_id(9));
	auto out_x_k_m  = out_x_k.map<float *>();
	if (out_x_k_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto out_Up = xrt::bo(device, DIM*DIM*sizeof(float), kalman_accelerator.group_id(10));
	auto out_Up_m  = out_Up.map<float *>();
	if (out_Up_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    }

	auto out_Dp = xrt::bo(device, DIM*sizeof(float), kalman_accelerator.group_id(11));
	auto out_Dp_m  = out_Dp.map<float *>();
	if (out_Dp_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    } 

	//create Buffer for communication with Hungarian Kernel 
	auto detections = xrt::bo(device, 150*sizeof(float), hungarian_accelerator.group_id(0));
	auto detections_m = detections.map<float *>();
	if (detections_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");   
    }

	auto kalman_filters = xrt::bo(device, 150*sizeof(float), hungarian_accelerator.group_id(1));
	auto kalman_filters_m = kalman_filters.map<float *>();
	if (kalman_filters_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");   
    }

	auto assignments = xrt::bo(device, 150*sizeof(int), hungarian_accelerator.group_id(2));
	auto assignments_m = assignments.map<int *>();
	if (assignments_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");   
    }

	auto ious = xrt::bo(device, 150*sizeof(float), hungarian_accelerator.group_id(3));
	auto ious_m = ious.map<float *>();
	if (ious_m == nullptr){
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");   
    }

    //map Pre-Post-Proc Kernel to handle 
    preproc_handle->kernel = std::move(preproc_accelerator);
	preproc_handle->runner = std::move(preproc_runner);
	preproc_handle->runner_align = std::move(alignmem_runner);
    preproc_handle->device = std::move(device);
    preproc_handle->data_in = std::move(img_preproc_in);
    preproc_handle->data_out = std::move(img_preproc_out);
    preproc_handle->data_in_m = img_preproc_in_m;
    preproc_handle->data_out_m = img_preproc_out_m;

    //map Post-Proc Kernel to handle 
    postproc_handle->kernel = std::move(postproc_accelerator);
	postproc_handle->runner = std::move(postproc_runner);
    postproc_handle->device = std::move(device);
	postproc_handle->items = std::move(items);
    postproc_handle->data_out = std::move(out_postproc);
	postproc_handle->items_m = items_m;
    postproc_handle->data_out_m = out_postproc_m; 

	//map Kalman Kernel to handle 
    kalman_handle->kernel = std::move(kalman_accelerator);
	kalman_handle->runner = std::move(kalman_runner);
    kalman_handle->device = std::move(device);
    kalman_handle->in_A = std::move(in_A);
	kalman_handle->in_Uq = std::move(in_Uq);
	kalman_handle->in_Dq = std::move(in_Dq);
    kalman_handle->in_H = std::move(in_H);
	kalman_handle->in_x_k1 = std::move(in_x_k1);
	kalman_handle->in_Up = std::move(in_Up);
    kalman_handle->in_Dp = std::move(in_Dp);
	kalman_handle->in_R = std::move(in_R);
	kalman_handle->in_z = std::move(in_z);
    kalman_handle->out_x_k = std::move(out_x_k);
	kalman_handle->out_Up = std::move(out_Up);
	kalman_handle->out_Dp = std::move(out_Dp);
	kalman_handle->in_A_m = in_A_m; 
	kalman_handle->in_Uq_m = in_Uq_m; 
	kalman_handle->in_Dq_m = in_Dq_m; 
	kalman_handle->in_x_k1_m = in_x_k1_m; 
	kalman_handle->in_Up_m = in_Up_m; 
	kalman_handle->in_Dp_m = in_Dp_m; 
	kalman_handle->in_R_m = in_R_m; 
	kalman_handle->in_z_m = in_z_m; 
	kalman_handle->in_H_m = in_H_m; 
	kalman_handle->out_x_k_m = out_x_k_m; 
	kalman_handle->out_Up_m = out_Up_m; 
	kalman_handle->out_Dp_m = out_Dp_m; 

	//initialize the static kalman filter matrices 
	KalmanFilter filter; 
	std::memcpy(kalman_handle->in_A_m, filter.A, DIM*DIM*sizeof(float));
	std::memcpy(kalman_handle->in_Uq_m, filter.Q, DIM*DIM*sizeof(float));
	std::memcpy(kalman_handle->in_Dq_m, filter.Dq, DIM*sizeof(float));
	std::memcpy(kalman_handle->in_R_m, filter.Dr, DIM*sizeof(float)/2);
	std::memcpy(kalman_handle->in_H_m, filter.H, 4*DIM*sizeof(float));

	kalman_handle->in_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	kalman_handle->in_Uq.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	kalman_handle->in_Dq.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	kalman_handle->in_R.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	kalman_handle->in_H.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	//map Hungarian Kernel to handle 
    hungarian_handle->kernel = std::move(hungarian_accelerator);
	hungarian_handle->runner = std::move(hungarian_runner);
    hungarian_handle->device = std::move(device);
    hungarian_handle->detections = std::move(detections);
	hungarian_handle->kalman_filters = std::move(kalman_filters);
	hungarian_handle->assignments = std::move(assignments);
    hungarian_handle->ious = std::move(ious);
	hungarian_handle->detections_m = detections_m; 
	hungarian_handle->kalman_filters_m = kalman_filters_m; 
	hungarian_handle->assignments_m = assignments_m; 
	hungarian_handle->ious_m = ious_m; 
}

/**
 * @brief invokeDPU 
 *
 * Function to invoke DPU and all other Kernels 
 *
 * @param *dpu_handle[in]: Handle of Pre-Processing Kernel 
 * @param *postproc_handle[in]: Handle of Post-Processing Kernel 
 * @param *kalman_handle[in]: Handle of Kalman Kernel 
 * @param *hungarian_handle[in]: Handle of Hungarian Kernel 
 * @param &img[in]: input image  
 * @param &img_resized[in]: output image  
 * @param &roi[in]: roi image to suppress detections 
 * @param *results_L[in]: results output tensor 13x13x3
 * @param *results_M[in]: results output tensor 26x26x3
 * @param *results_S[in]: results output tensor 52x52x3
 * @param *items[out]: number of detected objects  
 * @param *box_out[out]: bounding boxes of detected objects  
 * @param *cfg[in]: program configuration 
 * @param imname[in]: name of image to save the result 
 * @param obj_threshold[in]: object threshold   
 * @param nms_threshold[in]: threshold for non-maximum suppression  
 * @param &kalman_tracks[inout]: kalman hypotheses  
 * @param &trackings[inout]: saved kalman hypotheses 
 * @param *tracking_id[inout]: tracking id counter for new tracks to generate unique id   
 *  
 */
void invokeDPU(DPUKernelHandle *dpu_handle, PreKernelHandle *preproc_handle, PostKernelHandle *postproc_handle, KalmanKernelHandle *kalman_handle, HungarianHandle *hungarian_handle, cv::Mat &img, cv::Mat &img_resized, cv::Mat &roi, int8_t *results_L, int8_t *results_M, int8_t *results_S, int *items, float *box_out, ProgConfig *cfg, std::string imname, float obj_threshold, float nms_threshold, std::vector<KalmanFilter> &kalman_tracks, std::vector<Tracking> &trackings, int *tracking_id){
	
	/*Pre-Processing*/
	auto t_start = std::chrono::high_resolution_clock::now();
	if(cfg->processing == 0){
		/* Hardware pre-processing */
		invokePreKernel(preproc_handle, img.data, cfg->data_in_phy_addr); 
	} else if (cfg->processing == 1){
		/* Software postprocessing */
		preProcess(img, img_resized, cfg->img_in_width, cfg->img_in_height, cfg->img_out_width, cfg->img_out_height, preproc_handle->scale); 
		std::memcpy((int8_t *) cfg->data_in_addr, (int8_t *) img_resized.data, pre_out_size*sizeof(int8_t));
	} 
	auto t_end = std::chrono::high_resolution_clock::now();
	double time = std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cfg->timings_preproc[cfg->current_idx] = time; 

	/*DPU*/
	t_start = std::chrono::high_resolution_clock::now();
	// sync input tensor buffers
	for (auto& input : dpu_handle->input_tensor_buffers) {
		input->sync_for_write(0, input->get_tensor()->get_data_size() / input->get_tensor()->get_shape()[0]);
	}

	//run DPU 
	auto job_id = dpu_handle->dpu_runner->execute_async(dpu_handle->input_tensor_buffers, dpu_handle->output_tensor_buffers);
	dpu_handle->dpu_runner->wait((int)job_id.first, -1); 

	//sync output tensor buffers 
	for (auto& output : dpu_handle->output_tensor_buffers) {
		output->sync_for_read(0, output->get_tensor()->get_data_size() / output->get_tensor()->get_shape()[0]);
	}
	t_end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cfg->timings_dpu[cfg->current_idx] = time; 

	*items = 0; 

	/*Post-Processing*/
	t_start = std::chrono::high_resolution_clock::now();
	if (cfg->processing == 0){
		/* Hardware post-processing */
		invokePostKernel(postproc_handle, cfg->bboxes_L, cfg->bboxes_M, cfg->bboxes_S, obj_threshold, nms_threshold, cfg->data_out_phy_addr);
	} else if (cfg->processing == 1){
		/* Software post-processing */
		postProcess(results_L, results_M, results_S, postproc_handle->scale_large, postproc_handle->scale_medium, postproc_handle->scale_small, obj_threshold, nms_threshold, cfg->bboxes_L, cfg->bboxes_M, cfg->bboxes_S, items, box_out);
	} 
	t_end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cfg->timings_postproc[cfg->current_idx] = time;

	/* Object Tracking Processing */
	t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < trackings.size(); i++){
		trackings[i].is_tracked = false; 
	}
	//Step 1: Predict new states of hypotheses
	for (int i = 0; i < kalman_tracks.size(); i++){
		/* Software prediction*/
		if (cfg->processing == 1){
			float u[4] = {0, 0, 0, 0};
			kalman_tracks[i].is_tracked = false; 
			kalmanPredict(&kalman_tracks[i], u); 
		/* Hardware prediction*/
		} else if (cfg->processing == 0){
			kalman_tracks[i].is_tracked = false; 
			//copy data to kernel 
			std::memcpy(kalman_handle->in_x_k1_m, kalman_tracks[i].x, DIM*sizeof(float));
			std::memcpy(kalman_handle->in_Up_m, kalman_tracks[i].P, DIM*DIM*sizeof(float));
			std::memcpy(kalman_handle->in_Dp_m, kalman_tracks[i].Dp, DIM*sizeof(float));
			//sync data with kernel 
			kalman_handle->in_x_k1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			kalman_handle->in_Up.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			kalman_handle->in_Dp.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			//start runner 
			kalman_handle->runner(kalman_handle->in_A, kalman_handle->in_Uq, kalman_handle->in_Dq, kalman_handle->in_H, kalman_handle->in_x_k1, kalman_handle->in_Up, kalman_handle->in_Dp, kalman_handle->in_R, kalman_handle->in_z, kalman_handle->out_x_k, kalman_handle->out_Up, kalman_handle->out_Dp, predict_flag); 	
			kalman_handle->runner.wait();
			//sync data with kernel 
			kalman_handle->out_x_k.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			kalman_handle->out_Up.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			kalman_handle->out_Dp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			//copy data from kernel 
			std::memcpy(kalman_tracks[i].x, kalman_handle->out_x_k_m, DIM*sizeof(float));
			std::memcpy(kalman_tracks[i].P, kalman_handle->out_Up_m, DIM*DIM*sizeof(float));
			std::memcpy(kalman_tracks[i].Dp, kalman_handle->out_Dp_m, DIM*sizeof(float));
		}
	}

	//Step 2: Split data by object score  
	float box_high[*items*8] = {0}; 
	float box_low[*items*8] = {0}; 
	int high_counter = 0; 
	int low_counter = 0; 
	//iterate over all detections 
	for (int i = 0; i < *items; i++){
		//calculate box center for roi comparison 
		float height = box_out[i*8 + 3] - box_out[i*8 + 1];
		float y_center = box_out[i*8 + 1] + height/2; 
		float width = box_out[i*8 + 2] - box_out[i*8];
		float x_center = box_out[i*8] + width/2; 
		//compare box with roi 
		if (roi.data[(int(x_center)+int(cfg->img_out_width*y_center))*3] != 0){
			//list with high object scores 
			if (box_out[i*8+6] > OBJ_HIGH){
				for (int j = 0; j < 8; j++){
					box_high[high_counter*8+j] = box_out[i*8+j];
				}
				high_counter ++;
			//list with low object scores 
			} else {
				for (int j = 0; j < 8; j++){
					box_low[low_counter*8+j] = box_out[i*8+j];
				}
				low_counter ++;
			}
		}
	}

	//Step 3: Match high object score detections
	if (kalman_tracks.size() > 0){
		//matching step 1: create cost matrix and assign with hungarian algorithm 
		/* Software processing*/
		if (cfg->processing == 1){
			//create cost matrix 
			int matrix_dim = std::max(high_counter, (int)kalman_tracks.size()); 
			float ious_[matrix_dim*matrix_dim] = {0};
			float cost_matrix[matrix_dim*matrix_dim] = {0};
			//fill cost matrix with iou 
			for (int i = 0; i < high_counter; i++){
				for (int n = 0; n < kalman_tracks.size(); n++){
					float kalman_x1 = kalman_tracks[n].x[0] - kalman_tracks[n].x[2]/2;
					float kalman_x2 = kalman_tracks[n].x[0] + kalman_tracks[n].x[2]/2;
					float kalman_y1 = kalman_tracks[n].x[1] - kalman_tracks[n].x[3]/2;
					float kalman_y2 = kalman_tracks[n].x[1] + kalman_tracks[n].x[3]/2;
					float iou = calcIOU(box_high[i*8], box_high[i*8 + 2], box_high[i*8 + 1], box_high[i*8 + 3], kalman_x1, kalman_x2, kalman_y1, kalman_y2);
					ious_[i*matrix_dim+n] = iou;
					cost_matrix[i*matrix_dim+n] = 1 - iou; 
				}
			}
			//hungarian algorithm 
			int assignments[matrix_dim] = {-1};
			hungarianAlgorithm(cost_matrix, matrix_dim, (int *)assignments);
			for(int i = 0; i < matrix_dim; i++){
				int n = assignments[i]; 
				//match data and update hypothese if greater threshold 
				if ((n < high_counter) && (ious_[n*matrix_dim+i] > TRACKING_THRESHOLD) && (i < kalman_tracks.size())){
					kalman_tracks[i].lost_counter = 0; 
					kalman_tracks[i].is_tracked = true; 
					//calculate box information for update step 
					float height = box_high[n*8 + 3] - box_high[n*8 + 1];
					float y_center = box_high[n*8 + 1] + height/2; 
					float width = box_high[n*8 + 2] - box_high[n*8];
					float x_center = box_high[n*8] + width/2; 
					float z[4] = {x_center, y_center, width, height};
					//update 
					kalmanUpdate(&kalman_tracks[i], z);
					//calculate box information for original image size 
					float new_width = (box_high[n*8 + 2] - box_high[n*8]);
					float new_height = (box_high[n*8 + 3] - box_high[n*8 + 1]);
					float new_x_center = (box_high[n*8] + new_width/2)/cfg->img_out_width;
					float new_y_center = (box_high[n*8 + 1] + new_height/2 - 91)/234; 
					new_width = new_width/cfg->img_out_width; 
					new_height = new_height/234; 
					//save tracks 
					trackings[kalman_tracks[i].id].x.push_back((new_x_center)*cfg->img_in_width);
					trackings[kalman_tracks[i].id].y.push_back((new_y_center+new_height/2)*cfg->img_in_height);
					trackings[kalman_tracks[i].id].r = kalman_tracks[i].r;
					trackings[kalman_tracks[i].id].g = kalman_tracks[i].g;
					trackings[kalman_tracks[i].id].b = kalman_tracks[i].b;
					trackings[kalman_tracks[i].id].is_tracked = true; 
					box_high[n*8 + 7] = 1; 
					//update image and display hypothese  
					cv::putText(img,std::to_string(kalman_tracks[i].id),cv::Point(int((new_x_center-new_width/2)*cfg->img_in_width),int((new_y_center-new_height/2)*cfg->img_in_height+10)),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
					cv::rectangle(img, cv::Point(int((new_x_center-new_width/2)*cfg->img_in_width),int((new_y_center-new_height/2)*cfg->img_in_height)), cv::Point(int((new_x_center+new_width/2)*cfg->img_in_width),int((new_y_center+new_height/2)*cfg->img_in_height)), cv::Scalar(kalman_tracks[i].r, kalman_tracks[i].g, kalman_tracks[i].b), 2, cv::LINE_8);
				}
			}
		/* Hardware processing*/
		} else if (cfg->processing == 0){
			
			//copy box information to kernel 
			for (int i = 0; i < high_counter; i++){
				hungarian_handle->detections_m[i*4] = box_high[i*8]; 
				hungarian_handle->detections_m[i*4+1] = box_high[i*8+1]; 
				hungarian_handle->detections_m[i*4+2] = box_high[i*8+2]; 
				hungarian_handle->detections_m[i*4+3] = box_high[i*8+3]; 
			}

			//copy hypotheses information to kernel 
			for (int i = 0; i < kalman_tracks.size(); i++){
				float kalman_x1 = kalman_tracks[i].x[0] - kalman_tracks[i].x[2]/2;
				float kalman_x2 = kalman_tracks[i].x[0] + kalman_tracks[i].x[2]/2;
				float kalman_y1 = kalman_tracks[i].x[1] - kalman_tracks[i].x[3]/2;
				float kalman_y2 = kalman_tracks[i].x[1] + kalman_tracks[i].x[3]/2;
				hungarian_handle->kalman_filters_m[i*4] = kalman_x1; 
				hungarian_handle->kalman_filters_m[i*4+1] = kalman_y1; 
				hungarian_handle->kalman_filters_m[i*4+2] = kalman_x2; 
				hungarian_handle->kalman_filters_m[i*4+3] = kalman_y2; 
			}

			//get matrix dim 
			int matrix_dim = std::max(high_counter, (int)kalman_tracks.size()); 

			//invoke hungarian kernel 
			hungarian_handle->runner(hungarian_handle->detections, hungarian_handle->kalman_filters, hungarian_handle->assignments, hungarian_handle->ious, high_counter, (int)kalman_tracks.size()); 	
			hungarian_handle->runner.wait();

			for(int i = 0; i < matrix_dim; i++){
				int n = hungarian_handle->assignments_m[i]; 
				//match data and update hypothese if greater threshold 
				if ((n < high_counter) && (hungarian_handle->ious_m[n*matrix_dim+i] > TRACKING_THRESHOLD) && (i < kalman_tracks.size())){
					kalman_tracks[i].lost_counter = 0; 
					kalman_tracks[i].is_tracked = true; 
					//calculate box information for update step 
					float height = box_high[n*8 + 3] - box_high[n*8 + 1];
					float y_center = box_high[n*8 + 1] + height/2; 
					float width = box_high[n*8 + 2] - box_high[n*8];
					float x_center = box_high[n*8] + width/2; 
					float z[4] = {x_center, y_center, width, height};
					//copy data to kernel for update 
					std::memcpy(kalman_handle->in_x_k1_m, kalman_tracks[i].x, DIM*sizeof(float));
					std::memcpy(kalman_handle->in_Up_m, kalman_tracks[i].P, DIM*DIM*sizeof(float));
					std::memcpy(kalman_handle->in_Dp_m, kalman_tracks[i].Dp, DIM*sizeof(float));
					std::memcpy(kalman_handle->in_z_m, z, DIM*sizeof(float)/2);
					//sync data with kernel 
					kalman_handle->in_x_k1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
					kalman_handle->in_Up.sync(XCL_BO_SYNC_BO_TO_DEVICE);
					kalman_handle->in_Dp.sync(XCL_BO_SYNC_BO_TO_DEVICE);
					kalman_handle->in_z.sync(XCL_BO_SYNC_BO_TO_DEVICE);
					//start runner 
					kalman_handle->runner(kalman_handle->in_A, kalman_handle->in_Uq, kalman_handle->in_Dq, kalman_handle->in_H, kalman_handle->in_x_k1, kalman_handle->in_Up, kalman_handle->in_Dp, kalman_handle->in_R, kalman_handle->in_z, kalman_handle->out_x_k, kalman_handle->out_Up, kalman_handle->out_Dp, update_flag); 	
					kalman_handle->runner.wait();
					//sync data with kernel 
					kalman_handle->out_x_k.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
					kalman_handle->out_Up.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
					kalman_handle->out_Dp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
					//copy data from kernel 
					std::memcpy(kalman_tracks[i].x, kalman_handle->out_x_k_m, DIM*sizeof(float));
					std::memcpy(kalman_tracks[i].P, kalman_handle->out_Up_m, DIM*DIM*sizeof(float));
					std::memcpy(kalman_tracks[i].Dp, kalman_handle->out_Dp_m, DIM*sizeof(float));
					//calculate box information for original image size 
					float new_width = (box_high[n*8 + 2] - box_high[n*8]);
					float new_height = (box_high[n*8 + 3] - box_high[n*8 + 1]);
					float new_x_center = (box_high[n*8] + new_width/2)/cfg->img_out_width;
					float new_y_center = (box_high[n*8 + 1] + new_height/2 - 91)/234; 
					new_width = new_width/cfg->img_out_width; 
					new_height = new_height/234; 
					//save tracks
					trackings[kalman_tracks[i].id].x.push_back((new_x_center)*cfg->img_in_width);
					trackings[kalman_tracks[i].id].y.push_back((new_y_center+new_height/2)*cfg->img_in_height);
					trackings[kalman_tracks[i].id].r = kalman_tracks[i].r;
					trackings[kalman_tracks[i].id].g = kalman_tracks[i].g;
					trackings[kalman_tracks[i].id].b = kalman_tracks[i].b;
					trackings[kalman_tracks[i].id].is_tracked = true; 
					box_high[n*8 + 7] = 1; 
					//update image and display hypothese  
					cv::putText(img,std::to_string(kalman_tracks[i].id),cv::Point(int((new_x_center-new_width/2)*cfg->img_in_width),int((new_y_center-new_height/2)*cfg->img_in_height+10)),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
					cv::rectangle(img, cv::Point(int((new_x_center-new_width/2)*cfg->img_in_width),int((new_y_center-new_height/2)*cfg->img_in_height)), cv::Point(int((new_x_center+new_width/2)*cfg->img_in_width),int((new_y_center+new_height/2)*cfg->img_in_height)), cv::Scalar(kalman_tracks[i].r, kalman_tracks[i].g, kalman_tracks[i].b), 2, cv::LINE_8);
				}
			}
		}

		//Step 4: Match high object score detections
		for (int i = 0; i < low_counter; i++){
			for (int j = 0; j < kalman_tracks.size(); j++){
				if (kalman_tracks[j].is_tracked == false){
					//calculate hypotheses box information for iou 
					float kalman_x1 = kalman_tracks[j].x[0] - kalman_tracks[j].x[2]/2;
					float kalman_x2 = kalman_tracks[j].x[0] + kalman_tracks[j].x[2]/2;
					float kalman_y1 = kalman_tracks[j].x[1] - kalman_tracks[j].x[3]/2;
					float kalman_y2 = kalman_tracks[j].x[1] + kalman_tracks[j].x[3]/2;
					float iou = calcIOU(box_low[i*8], box_low[i*8 + 2], box_low[i*8 + 1], box_low[i*8 + 3], kalman_x1, kalman_x2, kalman_y1, kalman_y2);
					if (iou > TRACKING_THRESHOLD){
						kalman_tracks[j].lost_counter = 0; 
						kalman_tracks[j].is_tracked = true; 
						//calculate box information for update step 
						float height = box_low[i*8 + 3] - box_low[i*8 + 1];
						float y_center = box_low[i*8 + 1] + height/2; 
						float width = box_low[i*8 + 2] - box_low[i*8];
						float x_center = box_low[i*8] + width/2; 
						float z[4] = {x_center, y_center, width, height};
						/* Software processing*/
						if (cfg->processing == 1){
							kalmanUpdate(&kalman_tracks[j], z);
						/* Hardware processing*/
						} else if (cfg->processing == 0){
							//copy data to kernel 
							std::memcpy(kalman_handle->in_x_k1_m, kalman_tracks[j].x, DIM*sizeof(float));
							std::memcpy(kalman_handle->in_Up_m, kalman_tracks[j].P, DIM*DIM*sizeof(float));
							std::memcpy(kalman_handle->in_Dp_m, kalman_tracks[j].Dp, DIM*sizeof(float));
							std::memcpy(kalman_handle->in_z_m, z, DIM*sizeof(float)/2);
							//sync data with kernel 
							kalman_handle->in_x_k1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
							kalman_handle->in_Up.sync(XCL_BO_SYNC_BO_TO_DEVICE);
							kalman_handle->in_Dp.sync(XCL_BO_SYNC_BO_TO_DEVICE);
							kalman_handle->in_z.sync(XCL_BO_SYNC_BO_TO_DEVICE);
							//start runner 
							kalman_handle->runner(kalman_handle->in_A, kalman_handle->in_Uq, kalman_handle->in_Dq, kalman_handle->in_H, kalman_handle->in_x_k1, kalman_handle->in_Up, kalman_handle->in_Dp, kalman_handle->in_R, kalman_handle->in_z, kalman_handle->out_x_k, kalman_handle->out_Up, kalman_handle->out_Dp, update_flag); 	
							kalman_handle->runner.wait();
							//sync data with kernel 
							kalman_handle->out_x_k.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
							kalman_handle->out_Up.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
							kalman_handle->out_Dp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
							//copy data from kernel 
							std::memcpy(kalman_tracks[j].x, kalman_handle->out_x_k_m, DIM*sizeof(float));
							std::memcpy(kalman_tracks[j].P, kalman_handle->out_Up_m, DIM*DIM*sizeof(float));
							std::memcpy(kalman_tracks[j].Dp, kalman_handle->out_Dp_m, DIM*sizeof(float));
						} 		
						//calculate box information for original image size 
						float new_width = (box_low[i*8 + 2] - box_low[i*8]);
						float new_height = (box_low[i*8 + 3] - box_low[i*8 + 1]);
						float new_x_center = (box_low[i*8] + new_width/2)/cfg->img_out_width;
						float new_y_center = (box_low[i*8 + 1] + new_height/2 - 91)/234; 
						new_width = new_width/cfg->img_out_width; 
						new_height = new_height/234; 
						//save tracks 
						trackings[kalman_tracks[j].id].x.push_back((new_x_center)*cfg->img_in_width);
						trackings[kalman_tracks[j].id].y.push_back((new_y_center+new_height/2)*cfg->img_in_height);
						trackings[kalman_tracks[j].id].r = kalman_tracks[j].r;
						trackings[kalman_tracks[j].id].g = kalman_tracks[j].g;
						trackings[kalman_tracks[j].id].b = kalman_tracks[j].b;
						trackings[kalman_tracks[j].id].is_tracked = true; 
						box_low[i*8 + 7] = 1; 
						//update image and display hypothese  
						cv::putText(img,std::to_string(kalman_tracks[j].id),cv::Point(int((new_x_center-new_width/2)*cfg->img_in_width),int((new_y_center-new_height/2)*cfg->img_in_height+10)),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
						cv::rectangle(img, cv::Point(int((new_x_center-new_width/2)*cfg->img_in_width),int((new_y_center-new_height/2)*cfg->img_in_height)), cv::Point(int((new_x_center+new_width/2)*cfg->img_in_width),int((new_y_center+new_height/2)*cfg->img_in_height)), cv::Scalar(kalman_tracks[j].r, kalman_tracks[j].g, kalman_tracks[j].b), 2, cv::LINE_8);
					}
				}
			}
		}
	}

	//remove elements if not tracked for a certain amount of time  
	int remove_counter = 0;
	for (int i = 0; i < kalman_tracks.size(); i++){
		if (kalman_tracks[i-remove_counter].is_tracked == false){
			kalman_tracks[i-remove_counter].lost_counter = kalman_tracks[i-remove_counter].lost_counter + 1;
			if (kalman_tracks[i-remove_counter].lost_counter > 40){
				trackings[kalman_tracks[i-remove_counter].id].is_tracked = 0; 
				kalman_tracks.erase(kalman_tracks.begin()+i-remove_counter);
				remove_counter += 1; 
			}
		}
	}

	//initialize kalman filter for untracked objects 
	for (int i = 0; i < high_counter; i++){
		if (box_high[i*8 + 7] == 0){
			float height = box_high[i*8 + 3] - box_high[i*8 + 1];
			float y_center = box_high[i*8 + 1] + height/2; 
			float width = box_high[i*8 + 2] - box_high[i*8];
			float x_center = box_high[i*8] + width/2; 
			initializeKalman(kalman_tracks, x_center, y_center, width, height, *tracking_id); 
			*tracking_id += 1; 
			Tracking track; 
			trackings.push_back(track); 
		}
	}
	t_end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration<double, std::milli>(t_end-t_start).count();
	cfg->timings_kalman[cfg->current_idx] = time;

	cfg->current_idx = (cfg->current_idx+1) % 1000; 
}

/**
 * @brief invokePreKernel 
 *
 * Function to invoke Pre-Processing Kernel  
 *
 * @param *preproc_handle[inout]: Handle of Pre-Processing Kernel 
 * @param *data[in]: image data 
 * @param dpu_input_addr[in]: physical address of dpu input tensor 
 *  
 */
void invokePreKernel(PreKernelHandle *preproc_handle, unsigned char *data, uint64_t dpu_input_addr){
    //copy data to host side input of kernel 
    std::memcpy(preproc_handle->data_in_m, data, pre_in_size*sizeof(char));
    //sync data with kernel 
    preproc_handle->data_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    //start runner 
	preproc_handle->runner(preproc_handle->data_in, preproc_handle->data_out, preproc_handle->scale); 
	preproc_handle->runner.wait();
	preproc_handle->runner_align(preproc_handle->data_out, dpu_input_addr); 
	preproc_handle->runner_align.wait();
}

/**
 * @brief invokePostKernel 
 *
 * Function to invoke Post-Processing Kernel  
 *
 * @param *postproc_handle[inout]: Handle of Post-Processing Kernel 
 * @param *bboxes_L[in]: anchor boxes output tensor 13x13x3   
 * @param *bboxes_M[in]: anchor boxes output tensor 26x26x3      
 * @param *bboxes_S[in]: anchor boxes output tensor 52x52x3 
 * @param obj_threshold[in]: object threshold   
 * @param nms_threshold[in]: threshold for non-maximum suppression  
 * @param *adr[in]: physical address of dpu output tensors 
 *  
 */
void invokePostKernel(PostKernelHandle *postproc_handle, uint16_t *bboxes_L, uint16_t *bboxes_M, uint16_t *bboxes_S, float obj_threshold, float nms_threshold, uint64_t addr[3]){   
	//copy data to host side input of kernel 
	uint16_t __bboxes_L [6];
	uint16_t __bboxes_M [6];
	uint16_t __bboxes_S [6];
	for (int i = 0; i < 6; i ++) {
		__bboxes_L[i] = bboxes_L[i];
		__bboxes_M[i] = bboxes_M[i];
		__bboxes_S[i] = bboxes_S[i];
	}
	//start runner 
	postproc_handle->runner(addr[2], addr[1], addr[0], postproc_handle->items, postproc_handle->data_out, __bboxes_L, __bboxes_M, __bboxes_S, postproc_handle->scale_large, postproc_handle->scale_medium, postproc_handle->scale_small, obj_threshold, nms_threshold); 
	postproc_handle->runner.wait();
    //sync data with kernel 
    postproc_handle->data_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	postproc_handle->items.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}