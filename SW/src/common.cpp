#include "common.hpp"
#include "kernelhandle.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include "DPU.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <limits>
// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
//Vitis AI includes 
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
//Object Tracking Includes 
#include "kalman.hpp"

#define DIVIDER "----------------"

/**
 * @brief runDPU 
 *
 * Runner Function that executes the program and calls various functions for HW-Kernels 
 *
 * @param source[in]: source of the data {0=camera, 1=images}
 * @param processing[in]: type of processing {0=HW, 1=SW}
 * @param model[in]: path to compiled model 
 * @param dpu_path[in]: path to xclbin containing information about the hardware
 * @param img_path[in]: path to image directory 
 * @param roi_path[in]: path roi image to suppress detections in certain areas 
 *  
 */
int runDPU(uint8_t source, uint8_t processing, const char *model, const char *dpu_path, const char *img_path, const char *roi_path){

  //Initialize pre-, postprocessor, DPU Handle and config
  PreKernelHandle preproc_handle;
  PostKernelHandle postproc_handle;
  DPUKernelHandle dpu_handle; 
  KalmanKernelHandle kalman_handle; 
  HungarianHandle hungarian_handle; 
  ProgConfig cfg; 
  std::vector<KalmanFilter> kalman_tracks;
  std::vector<Tracking> trackings; 
  int tracking_id = 0; 

  //initialize configuration structure 
  cfg.grid_scale_L = 416/13; 
  cfg.grid_scale_M = 416/26;
  cfg.grid_scale_S = 416/52;
  cfg.max_boxes = MAX_PROPOSALS; 
  cfg.img_in_width = 1280;
  cfg.img_in_height = 720; 
  cfg.img_out_width = 416; 
  cfg.img_out_height = 416; 
  cfg.processing = processing; 
  cfg.source = source; 

  //allocate data for postprocessing --> wrap box read in function 
  cfg.bboxes_L = (uint16_t *)malloc(6 * sizeof(cfg.bboxes_L));
  cfg.bboxes_L[0] = 116; 
  cfg.bboxes_L[1] = 90; 
  cfg.bboxes_L[2] = 156; 
  cfg.bboxes_L[3] = 198; 
  cfg.bboxes_L[4] = 373; 
  cfg.bboxes_L[5] = 326; 
  cfg.bboxes_M = (uint16_t *)malloc(6 * sizeof(cfg.bboxes_M));
  cfg.bboxes_M[0] = 30; 
  cfg.bboxes_M[1] = 61; 
  cfg.bboxes_M[2] = 62; 
  cfg.bboxes_M[3] = 45; 
  cfg.bboxes_M[4] = 59; 
  cfg.bboxes_M[5] = 119;
  cfg.bboxes_S = (uint16_t *)malloc(6 * sizeof(cfg.bboxes_S));
  cfg.bboxes_S[0] = 10; 
  cfg.bboxes_S[1] = 13; 
  cfg.bboxes_S[2] = 16; 
  cfg.bboxes_S[3] = 30; 
  cfg.bboxes_S[4] = 33; 
  cfg.bboxes_S[5] = 23;

  //allocate box_out 
  float *box_out;
  box_out = (float *)malloc(cfg.max_boxes * 8 * sizeof(box_out));
  int *items;
  items = (int *)malloc(sizeof(items));

  //initialize DPU Kernel  
  std::cout << DIVIDER << "\n" << "Initializing DPU Kernel" << std::endl; 
	//deserialize graph and get subgraph for dpu
	auto graph = xir::Graph::deserialize(model);
	auto subgraph = get_dpu_subgraph(graph.get());
	//create dpu runner 
	auto attrs = xir::Attrs::create();
	auto dpu_runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
	//read input and output tensors 
	auto input_tensor_buffers = dpu_runner->get_inputs();
	auto output_tensor_buffers = dpu_runner->get_outputs();

	//write to handle struct
	dpu_handle.dpu_runner = std::move(dpu_runner);
	dpu_handle.input_tensor_buffers = std::move(input_tensor_buffers);
  dpu_handle.output_tensor_buffers = std::move(output_tensor_buffers);

  //read scaling information for pre- and postprocessor Kernel
  auto input_scale = vart::get_input_scale(dpu_handle.dpu_runner->get_input_tensors());
  auto output_scale = vart::get_output_scale(dpu_handle.dpu_runner->get_output_tensors());

  //define scalings 
  //output scaling as float, input scaling as uint8
  preproc_handle.scale = (unsigned char)input_scale[0];
  postproc_handle.scale_small = output_scale[0];
  postproc_handle.scale_medium = output_scale[1];
  postproc_handle.scale_large = output_scale[2];

  std::cout << "DPU Kernel Configuration finished\n" << DIVIDER << std::endl;

  //create cv Mat buffer
  cv::Mat img(cfg.img_in_width, cfg.img_in_height, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat img_resized(cfg.img_out_width, cfg.img_out_height, CV_8UC1, cv::Scalar(0, 0, 0));
  cv::Mat roi(cfg.img_out_width, cfg.img_out_height, CV_8UC3, cv::Scalar(0, 0, 0));
  roi = cv::imread(roi_path); 

  cv::namedWindow("YoloV7 Object Tracking @ Xilinx ZCU102",cv::WINDOW_FULLSCREEN);

  //do some additional initialization for Hardware pre- and postprocessing
  if(processing == 0){
    //Initialize pre- and postprocessor Kernel 
    std::cout << "Initializing pre- and postprocessor Kernel" << std::endl; 
    prepostKernelInit(&preproc_handle, &postproc_handle, &kalman_handle, &hungarian_handle, dpu_path);
    std::cout << "Kernel Configuration finished\n" << DIVIDER << std::endl;
    //map img output buffer 
    img_resized.data = preproc_handle.data_out_m;
    //map boxes and items to output of postprocess kernel 
    box_out = postproc_handle.data_out_m;
    items = postproc_handle.items_m;
  }

  //get DPU Kernel input buffer 
  uint64_t dpu_input_size = 0u;
  uint64_t data_in_addr;
  std::tie(data_in_addr, dpu_input_size) = dpu_handle.input_tensor_buffers[0]->data_phy({0, 0, 0, 0});
  cfg.data_in_phy_addr = data_in_addr; 
  std::tie(data_in_addr, dpu_input_size) = dpu_handle.input_tensor_buffers[0]->data({0, 0, 0, 0});
  cfg.data_in_addr = data_in_addr; 
  //get DPU Kernel output buffer
  uint64_t dpu_output_size = 0u; 
  uint64_t data_out_addr; 
  std::tie(data_out_addr, dpu_output_size) = dpu_handle.output_tensor_buffers[0]->data_phy({0, 0, 0, 0});
  cfg.data_out_phy_addr[0] = data_out_addr; 
  std::tie(data_out_addr, dpu_output_size) = dpu_handle.output_tensor_buffers[1]->data_phy({0, 0, 0, 0});
  cfg.data_out_phy_addr[1] = data_out_addr; 
  std::tie(data_out_addr, dpu_output_size) = dpu_handle.output_tensor_buffers[2]->data_phy({0, 0, 0, 0});
  cfg.data_out_phy_addr[2] = data_out_addr; 
  auto out_ptr_L = dpu_handle.output_tensor_buffers[2]->data({0, 0, 0, 0});
  auto out_ptr_M = dpu_handle.output_tensor_buffers[1]->data({0, 0, 0, 0});
  auto out_ptr_S = dpu_handle.output_tensor_buffers[0]->data({0, 0, 0, 0});
  int8_t *results_L = (int8_t *)out_ptr_L.first;
  int8_t *results_M = (int8_t *)out_ptr_M.first;
  int8_t *results_S = (int8_t *)out_ptr_S.first;

  if(cfg.source == 0){
    std::cout << "Starting inference with USB Camera" << std::endl;
    //live USB-Camera inference
    //create VideoCapture
    cv::VideoCapture cap;
    cap.open(0);
    // check if open succeeded
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //set frame height and width 
    cap.set(cv::CAP_PROP_FRAME_WIDTH,cfg.img_in_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,cfg.img_in_height);

    for (;;){
      //read frame from camera 
      cap.read(img);
      // check if read was good
      if (img.empty()) {
          std::cerr << "ERROR! blank frame grabbed\n";
          return 1;
      }
      //start inference and output data 
      auto t_start = std::chrono::high_resolution_clock::now();
      invokeDPU(&dpu_handle, &preproc_handle, &postproc_handle, &kalman_handle, &hungarian_handle, img, img_resized, roi, results_L, results_M, results_S, items, box_out, &cfg, "", OBJ_THRESHOLD, NMS_THRESHOLD, kalman_tracks, trackings, &tracking_id);
      auto t_end = std::chrono::high_resolution_clock::now();
	    auto time = std::chrono::duration<double, std::milli>(t_end-t_start).count();
	    cfg.timings_total[cfg.current_idx] = time;
      cv::imshow("YoloV7 Object Tracking @ Xilinx ZCU102", img);
      if (cv::waitKey(21) == 27){
          std::cout << "Inference stopped\n" << DIVIDER << std::endl;
          cap.release();
          break;
      }
    } 
  } else if (cfg.source == 1){
    int count = 0;
    //count files in directory 
    std::filesystem::path p1 { img_path };
    for (auto& p : std::filesystem::directory_iterator(p1))
    {
      ++count;
    }
    //inference with images from directory 
    std::cout << "Reading all images from " << img_path << " .." << std::endl; 
    std::ofstream outfile("det.txt");
    for (int i = 0; i < count; i++){
      //start inference and output data 
      std::string img_name = img_path + std::to_string(i) + ".jpg"; 
      img = cv::imread(img_name); 
      auto t_start = std::chrono::high_resolution_clock::now();
      invokeDPU(&dpu_handle, &preproc_handle, &postproc_handle, &kalman_handle, &hungarian_handle, img, img_resized, roi, results_L, results_M, results_S, items, box_out, &cfg, "", OBJ_THRESHOLD, NMS_THRESHOLD, kalman_tracks, trackings, &tracking_id);
      auto t_end = std::chrono::high_resolution_clock::now();
	    auto time = std::chrono::duration<double, std::milli>(t_end-t_start).count();
	    cfg.timings_total[cfg.current_idx] = time;
      cv::imshow("YoloV7 Object Tracking @ Xilinx ZCU102", img);
      if (cv::waitKey(1) == 27){
          std::cout << "Inference stopped\n" << DIVIDER << std::endl;
          break;
      }   
      //write all active hypotheses to detection file 
      for (int j = 0; j < kalman_tracks.size(); j++){
        if (kalman_tracks[j].is_tracked == true){
						float new_width = kalman_tracks[j].x[2]/416;
						float new_height = kalman_tracks[j].x[3]/234;
						float new_x_center = kalman_tracks[j].x[0]/416;
						float new_y_center = (kalman_tracks[j].x[1]-91)/234; 
            outfile << i << "," << kalman_tracks[j].id << "," << new_x_center*960 - new_width*960/2 << "," << new_y_center*540 - new_height*540/2 << "," << new_width*960 << "," << new_height*540 << "," << 1 << "," << -1 << "," << -1 << "," << -1 << std::endl;
        }
      }
    }
    outfile.close();
  }
  cv::destroyWindow("YoloV7 Object Tracking @ Xilinx ZCU102");
  std::cout << "total of " << tracking_id << " tracks." << std::endl; 
  //compute timings 
  computeTimings(cfg);
  return 0; 
}

/**
 * @brief preProcess 
 *
 * Software implementation of Pre-Processing
 *
 * @param src[in]: input image
 * @param dst[out]: output image 
 * @param in_width[in]: width of input image
 * @param in_height[in]: height of input image 
 * @param out_width[in]: width of output image 
 * @param out_height[in]: height of output image  
 * @param scaling_factor[in]: dpu input scaling factor  
 *  
 */
void preProcess(const cv::Mat &src, cv::Mat &dst, int in_width, int in_height, int out_width, int out_height, unsigned char scaling_factor){
  //calculate ratio and new height
  float ratio = (float)in_width/(float)in_height; 
  float new_height = (float)out_width/ratio;
  cv::Mat tmp; 
  cv::Mat tmp_cvt; 
  //resize 
  cv::resize(src, tmp, cv::Size(out_width, (int)new_height), cv::INTER_LINEAR);
  //cvt Color 
  cv::cvtColor(tmp, tmp_cvt, cv::COLOR_BGR2RGB);
  //make Border 
  int padding = dst.rows - tmp_cvt.rows;  
  cv::copyMakeBorder(tmp_cvt, dst, (padding/2), (padding/2), 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
  //data scaling with scaling factor of DPU input layer 
  float temp = 0; 
  for (int i = 0; i < out_height*out_width*3; i++){
    temp = float(dst.data[i])/255.0;
    temp = temp * (float)scaling_factor; 
    dst.data[i] = (unsigned char)temp; 
  }
}

/**
 * @brief postProcess 
 *
 * Software implementation of Post-Processing
 *
 * @param *results_L[in]: results output tensor 13x13x3
 * @param *results_M[in]: results output tensor 26x26x3
 * @param *results_S[in]: results output tensor 52x52x3
 * @param data_scale_L[in]: dpu output tensor scaling factor 13x13x3
 * @param data_scale_M[in]: dpu output tensor scaling factor 26x26x3 
 * @param data_scale_S[in]: dpu output tensor scaling factor 52x52x3  
 * @param obj_threshold[in]: object threshold   
 * @param nms_threshold[in]: threshold for non-maximum suppression  
 * @param *bboxes_L[in]: anchor boxes output tensor 13x13x3   
 * @param *bboxes_M[in]: anchor boxes output tensor 26x26x3      
 * @param *bboxes_S[in]: anchor boxes output tensor 52x52x3     
 * @param *items_out[out]: number of detected objects  
 * @param *box_out[out]: bounding boxes of detected objects  
 *  
 */
void postProcess(int8_t *results_L, int8_t *results_M, int8_t *results_S, float data_scale_L, float data_scale_M, float data_scale_S, float obj_threshold, float nms_threshold, uint16_t *bboxes_L, uint16_t *bboxes_M, uint16_t *bboxes_S, int *items_out, float *box_out){
	//allocate temporary memory for proposals 
  float arr_obj_scores [MAX_PROPOSALS] = {0};
	float arr_x1 [MAX_PROPOSALS];
	float arr_x2 [MAX_PROPOSALS];
	float arr_y1 [MAX_PROPOSALS];
	float arr_y2 [MAX_PROPOSALS];
	float arr_class_prob [MAX_PROPOSALS];
	float arr_detected_class [MAX_PROPOSALS];

  int feature_size = 5+N_CLASSES; 
  int num_proposals = 0; 

  //post process output tensor size 13x13x3
  for (int r = 0; r < TENSOR_SIZE_L; r++){
    for (int c = 0; c < TENSOR_SIZE_L; c++){
      for (int anchor = 0; anchor < 3; anchor++){
        float obj_score = 1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size+4])*data_scale_L));
        if(obj_score > obj_threshold && num_proposals < MAX_PROPOSALS){
          //x_center
          float x_center = (1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size])*data_scale_L))*2.0 - 0.5 + c)*GRID_SCALE_L;
          //y_center
          float y_center = (1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size+1])*data_scale_L))*2.0 - 0.5 + r)*GRID_SCALE_L; 
          //width
          float width = std::pow((1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size+2])*data_scale_L)))*2,2)*bboxes_L[2*anchor+0];
          //height
          float height = std::pow((1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size+3])*data_scale_L)))*2,2)*bboxes_L[2*anchor+1]; 
          //x1,y1,x2,y2 of bbox
          arr_x1[num_proposals] = (x_center - width/2);
          arr_y1[num_proposals] = (y_center - height/2);
          arr_x2[num_proposals] = (x_center + width/2);
          arr_y2[num_proposals] = (y_center + height/2);
          arr_obj_scores[num_proposals] = obj_score;
          arr_class_prob[num_proposals] = 1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size+5])*data_scale_L));
          arr_detected_class[num_proposals] = 0;
          //class probabilities
          for(int j=6; j < feature_size; j++){
            float tmp_score = 1/(1+std::exp(-1*float(results_L[(r*TENSOR_SIZE_L*3+c*3+anchor)*feature_size+j])*data_scale_L));
            if (arr_class_prob[num_proposals] < tmp_score) {
              arr_class_prob[num_proposals] = tmp_score;
              arr_detected_class[num_proposals] = j-5;
            }

          }
          num_proposals ++;
        }
      }
    }
  }

  //post process output tensor 26x26x3
  for (int r = 0; r < TENSOR_SIZE_M; r++){
    for (int c = 0; c < TENSOR_SIZE_M; c++){
      for (int anchor = 0; anchor < 3; anchor++){
        float obj_score = 1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size+4])*data_scale_M));
        if(obj_score > obj_threshold && num_proposals < MAX_PROPOSALS){
          //x_center
          float x_center = (1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size])*data_scale_M))*2.0 - 0.5 + c)*GRID_SCALE_M;
          //y_center
          float y_center = (1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size+1])*data_scale_M))*2.0 - 0.5 + r)*GRID_SCALE_M; 
          //width
          float width = std::pow((1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size+2])*data_scale_M)))*2,2)*bboxes_M[2*anchor+0];
          //height
          float height = std::pow((1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size+3])*data_scale_M)))*2,2)*bboxes_M[2*anchor+1]; 
          //x1,y1,x2,y2 of bbox
          arr_x1[num_proposals] = (x_center - width/2);
          arr_y1[num_proposals] = (y_center - height/2);
          arr_x2[num_proposals] = (x_center + width/2);
          arr_y2[num_proposals] = (y_center + height/2);
          arr_obj_scores[num_proposals] = obj_score;
          arr_class_prob[num_proposals] = 1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size+5])*data_scale_M));
          arr_detected_class[num_proposals] = 0;
          //class probabilities
          for(int j=6; j < feature_size; j++){
            float tmp_score = 1/(1+std::exp(-1*float(results_M[(r*TENSOR_SIZE_M*3+c*3+anchor)*feature_size+j])*data_scale_M));
            if (arr_class_prob[num_proposals] < tmp_score) {
              arr_class_prob[num_proposals] = tmp_score;
              arr_detected_class[num_proposals] = j-5;
            }
          }
          num_proposals ++;
        }
      }
    }
  }

  //post process output tensor 52x52x3
  for (int r = 0; r < TENSOR_SIZE_S; r++){
    for (int c = 0; c < TENSOR_SIZE_S; c++){
      for (int anchor = 0; anchor < 3; anchor++){
        float obj_score = 1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size+4])*data_scale_S));
        if(obj_score > obj_threshold && num_proposals < MAX_PROPOSALS){
          //x_center
          float x_center = (1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size])*data_scale_S))*2.0 - 0.5 + c)*GRID_SCALE_S;
          //y center 
          float y_center = (1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size+1])*data_scale_S))*2.0 - 0.5 + r)*GRID_SCALE_S; 
          //width
          float width = std::pow((1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size+2])*data_scale_S)))*2,2)*bboxes_S[2*anchor+0];
          //height
          float height = std::pow((1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size+3])*data_scale_S)))*2,2)*bboxes_S[2*anchor+1]; 
          //x1,y1,x2,y2 of bbox
          arr_x1[num_proposals] = (x_center - width/2);
          arr_y1[num_proposals] = (y_center - height/2);
          arr_x2[num_proposals] = (x_center + width/2);
          arr_y2[num_proposals] = (y_center + height/2);
          arr_obj_scores[num_proposals] = obj_score;
          arr_class_prob[num_proposals] = 1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size+5])*data_scale_S));
          arr_detected_class[num_proposals] = 0;
          //class probabilities
          for(int j=6; j < feature_size; j++){
            float tmp_score = 1/(1+std::exp(-1*float(results_S[(r*TENSOR_SIZE_S*3+c*3+anchor)*feature_size+j])*data_scale_S));
            if (arr_class_prob[num_proposals] < tmp_score) {
              arr_class_prob[num_proposals] = tmp_score;
              arr_detected_class[num_proposals] = j-5;
            }

          }
          num_proposals ++;
        }
      }
    }
  }

  //sort data by object score in descending order 
  for (int i = 0; i < num_proposals-1; i++) {
    // Last i elements are already in place
    for (int j = 0; j < num_proposals-i-1; j++) {
      // Swap if the element found is greater than the next element
      if (arr_obj_scores[j] < arr_obj_scores[j+1]) {
          float temp = arr_obj_scores[j];
          arr_obj_scores[j] = arr_obj_scores[j+1];
          arr_obj_scores[j+1] = temp;
          temp = arr_x1[j];
          arr_x1[j] = arr_x1[j+1];
          arr_x1[j+1] = temp;
          temp = arr_x2[j];
          arr_x2[j] = arr_x2[j+1];
          arr_x2[j+1] = temp;
          temp = arr_y1[j];
          arr_y1[j] = arr_y1[j+1];
          arr_y1[j+1] = temp;
          temp = arr_y2[j];
          arr_y2[j] = arr_y2[j+1];
          arr_y2[j+1] = temp;
          temp = arr_class_prob[j];
          arr_class_prob[j] = arr_class_prob[j+1];
          arr_class_prob[j+1] = temp;
          temp = arr_detected_class[j];
          arr_detected_class[j] = arr_detected_class[j+1];
          arr_detected_class[j+1] = temp;
      }
    }
  }

	int current_idx = 0;
	//nms algorithm 
	for (int i = 0; i < num_proposals; i++){
		if (arr_obj_scores[i] == 0){
			continue;
		}
		box_out[current_idx*8+0] = arr_x1[i];
		box_out[current_idx*8+1] = arr_y1[i];
		box_out[current_idx*8+2] = arr_x2[i];
		box_out[current_idx*8+3] = arr_y2[i];
		box_out[current_idx*8+4] = arr_detected_class[i];
		box_out[current_idx*8+5] = arr_class_prob[i];
		box_out[current_idx*8+6] = arr_obj_scores[i];
		box_out[current_idx*8+7] = 0;
		arr_obj_scores[i] = 0;
    //calculate iou against all other proposals 
		for (int j = i+1; j < num_proposals; j++){
			if (arr_obj_scores[j] == 0){
				continue;
			}
			if (calcIOU(arr_x1[i], arr_x2[i], arr_y1[i], arr_y2[i], arr_x1[j], arr_x2[j], arr_y1[j], arr_y2[j]) > NMS_THRESHOLD){
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

/**
 * @brief calcIOU 
 *
 * Software implementation of Intersection over Union 
 *
 * @param c_x1[in]: upper left x coordinate of candidate 
 * @param c_x2[in]: upper left y coordinate of candidate
 * @param c_y1[in]: lower right x coordinate of candidate 
 * @param c_y2[in]: lower right y coordinate of candidate
 * @param p_x1[in]: upper left x coordinate of proposal 
 * @param p_x2[in]: upper left y coordinate of proposal
 * @param p_y1[in]: lower right x coordinate of proposal 
 * @param p_y2[in]: lower right y coordinate of proposal  
 * @param iou[out]: intersection over union of candidate and proposal 
 *  
 */
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

/**
 * @brief computeTimings 
 *
 * Function to compute timings  
 *
 * @param cfg[in]: configuration struct that contains all recorded timings  
 *  
 */
void computeTimings(ProgConfig cfg){
  //preprocessing time 
  std::cout << "Calculating average computing time.." << std::endl;
  int i = 0; 
  double pre_time = 0; 
  double post_time = 0; 
  double dpu_time = 0; 
  double kalman_time = 0; 
  double total_time = 0; 
  for (i = 0; i < 1000; i++){
    if (cfg.timings_preproc[i] != 0){
      pre_time += cfg.timings_preproc[i];
      post_time += cfg.timings_postproc[i];
      dpu_time += cfg.timings_dpu[i];
      kalman_time += cfg.timings_kalman[i];
      total_time += cfg.timings_total[i];
    } else {
      pre_time = pre_time / (i); 
      dpu_time = dpu_time / (i); 
      post_time = post_time / (i); 
      kalman_time = kalman_time / (i); 
      total_time = total_time / (i); 
      break; 
    }
    if (i==999) {
      pre_time = pre_time / 1000;
      dpu_time = dpu_time / 1000; 
      post_time = post_time / 1000; 
      kalman_time = kalman_time / 1000; 
      total_time = total_time / 1000; 
    }
  } 

  std::cout << "Preprocessing took " << pre_time << " ms in avg." << std::endl;
  std::cout << "DPU took " << dpu_time << " ms in avg." << std::endl;
  std::cout << "Postprocessing took " << post_time << " ms in avg." << std::endl;
  std::cout << "Kalman Tracking took " << kalman_time << " ms in avg.\n" << DIVIDER << std::endl;
  std::cout << "Total Processing took " << total_time << " ms in avg.\n" << DIVIDER << std::endl;
}

/**
 * @brief hungarianAlgorithm  
 *
 * Software implementation of the hungarian algorithm   
 *
 * @param *cost_matrix[in]: iou cost matrix   
 * @param num_jobs[in]: size of the cost matrix 
 * @param assignments[out]: assignment vector 
 *  
 */
void hungarianAlgorithm(float* cost_matrix, int num_jobs, int* assignments) {

    // subtract the row minimum from each row
    for (int i = 0; i < num_jobs; i++) {
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

    std::vector<bool> row_covered(num_jobs, false);
    std::vector<bool> col_covered(num_jobs, false);
    int num_assigned = 0;

    // assign detections to kalman boxes
    while (num_assigned < num_jobs) {
        //assign zeros 
        for (int i = 0; i < num_jobs; i++) {
            if (row_covered[i]) {
                continue;
            }
            for (int j = 0; j < num_jobs; j++) {
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

        //get the minimum uncovered value in cost matrix 
        float min_uncovered = std::numeric_limits<float>::max();
        for (int i = 0; i < num_jobs; i++) {
            if (row_covered[i]) {
                continue;
            }
            for (int j = 0; j < num_jobs; j++) {
                if (col_covered[j]) {
                    continue;
                }
                min_uncovered = std::min(min_uncovered, cost_matrix[i * num_jobs + j]);
            }
        }

        // add the minimum uncovered value to the covered rows
        for (int i = 0; i < num_jobs; i++) {
            if (row_covered[i]) {
                for (int j = 0; j < num_jobs; j++) {
                    cost_matrix[i * num_jobs + j] += min_uncovered;
                }
            }
        }

        // subtract the minimum uncovered value from the uncovered columns
        for (int j = 0; j < num_jobs; j++) {
            if (!col_covered[j]) {
                for (int i = 0; i < num_jobs; i++) {
                    cost_matrix[i * num_jobs + j] -= min_uncovered;
                }
            }
        }
    }
}

