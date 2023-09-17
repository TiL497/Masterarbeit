#include "kalman.hpp"

//Function to multiply two matrices 
void matMul(float *mat_A, float *mat_B, float *result, int rows_A, int cols_A, int rows_B, int cols_B);
//Function to add two matrices 
void matAdd(float *mat_A, float *mat_B, float *result, int rows, int cols);
//Function to subtract two matrices 
void matSub(float *mat_A, float *mat_B, float *result, int rows, int cols);
//Function to transpose a Matrix  
void matTranspose(float *mat_A, float *result, int rows, int cols); 
//Function to calculate the inverse of a 4x4 Matrix 
void matInverse(float *mat, float *inverse); 