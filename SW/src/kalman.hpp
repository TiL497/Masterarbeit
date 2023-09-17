#pragma once 
#include <cstdint>
#include <cmath>
#include <vector>

//definitions for Kalman 
#define DIM 8
#define DT 0.033f
#define VAR_ACC std::pow(10,2) 
#define VAR_X   std::pow(0.1,2)
#define VAR_Y   std::pow(0.1,2)
#define VAR_W   std::pow(0.01,2)
#define VAR_H   std::pow(0.01,2)

//Struct to save tracks 
typedef struct Tracking{
    std::vector<float> x; 
    std::vector<float> y; 
    bool is_tracked = 1; 
    uint8_t r = 0; 
    uint8_t g = 0; 
    uint8_t b = 0; 
} Tracking;

//Struct for Kalman Hypothese
typedef struct KalmanFilter{
    //State of tracking 
    int id = 0; 
    bool is_tracked = false; 
    uint16_t lost_counter = 0;  
    uint8_t r = 0; 
    uint8_t g = 0; 
    uint8_t b = 0; 
    //State Vector 
    float x[DIM]     = {0}; 
    //Define the State Transition Matrix A
    float A[DIM*DIM] = {1 ,0 ,0 ,0 ,DT,0 ,0 ,0 ,
                        0 ,1 ,0 ,0 ,0 ,DT,0 ,0 ,
                        0 ,0 ,1 ,0 ,0 ,0 ,DT,0 ,
                        0 ,0 ,0 ,1 ,0 ,0 ,0 ,DT,
                        0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 };
    //Define the Control Input Matrix B
    float B[4*DIM]   = {0.5 * std::pow(DT,2),0                   ,0                   ,0                   , 
                        0                   ,0.5 * std::pow(DT,2),0                   ,0                   ,
                        0                   ,0                   ,0.5 * std::pow(DT,2),0                   ,
                        0                   ,0                   ,0                   ,0.5 * std::pow(DT,2), 
                        DT                  ,0                   ,0                   ,0                   , 
                        0                   ,DT                  ,0                   ,0                   ,
                        0                   ,0                   ,DT                  ,0                   ,
                        0                   ,0                   ,0                   ,DT                  };
    //Define Measurement Mapping Matrix
    float H[4*DIM]   = {1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 }; 
    float Q[DIM*DIM] = {std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,
                        0                        ,std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,
                        0                        ,0                        ,std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,
                        0                        ,0                        ,0                        ,std::pow(DT,4)*VAR_ACC/4 ,0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,
                        std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   ,0                        ,0                        ,0                        ,
                        0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   ,0                        ,0                        ,
                        0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   ,0                        ,
                        0                        ,0                        ,0                        ,std::pow(DT,3)*VAR_ACC/2 ,0                        ,0                        ,0                        ,std::pow(DT,2)*VAR_ACC   }; 
	float Dq[DIM]    = {std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,4)*VAR_ACC/4 ,std::pow(DT,2)*VAR_ACC ,std::pow(DT,2)*VAR_ACC ,std::pow(DT,2)*VAR_ACC ,std::pow(DT,2)*VAR_ACC};
    //Initial Measurement Noise Covariance
    float R[4*4]     = {VAR_X ,0     ,0     ,0    ,
                        0     ,VAR_Y ,0     ,0    ,
                        0     ,0     ,VAR_W ,0    ,
                        0     ,0     ,0     ,VAR_H};
    float Dr[4]      = {VAR_X ,VAR_Y ,VAR_W ,VAR_H};
    //Initial Covariance Matrix
    float P[DIM*DIM] = {1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 };
    float Dp[DIM]    = {1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 };
    //Identity Matrix
    float I[DIM*DIM] = {1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,
                        0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 };
} KalmanFilter;

//Function to initialize new kalman hypothese 
void initializeKalman(std::vector<KalmanFilter> &kalman_tracks, float x, float y, float w, float h, int id);
//Function to predict new state of kalman hypothese 
void kalmanPredict(KalmanFilter *kalman, float *u);
//Function to update kalman hypothese 
void kalmanUpdate(KalmanFilter *kalman, float *z); 