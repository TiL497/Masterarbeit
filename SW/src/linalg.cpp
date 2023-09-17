#include "linalg.hpp"
#include <bits/stdc++.h>

/**
 * @brief matMul 
 *
 * Function to multiply two matrices 
 *
 * @param *mat_A[in]: Matrix A   
 * @param *mat_A[in]: Matrix B  
 * @param *result[out]: result   
 * @param rows_A[in]: number of rows Matrix A   
 * @param cols_A[in]: number of columns Matrix A   
 * @param rows_B[in]: number of rows Matrix B   
 * @param cols_B[in]: nubmer of columns Matrix B   
 *  
 */
void matMul(float *mat_A, float *mat_B, float *result, int rows_A, int cols_A, int rows_B, int cols_B){

	float temp_result = 0;
	//fixing overwrite error 
	float temp_mat_B[rows_B*cols_B] = {0};
	float temp_mat_A[rows_A*cols_A] = {0}; 
	std::memcpy(temp_mat_B, mat_B, sizeof(temp_mat_B));
	std::memcpy(temp_mat_A, mat_A, sizeof(temp_mat_A));

	for (int i = 0; i < rows_A; i++){
		for (int j = 0; j < cols_B; j++){
			temp_result = 0.0f;
			for (int k = 0; k < cols_A; k++){
				temp_result = temp_result + temp_mat_A[i*cols_A+k] * temp_mat_B[k*cols_B+j];
			} 
			result[i*cols_B+j] = temp_result;
		}
	}
}

/**
 * @brief matAdd 
 *
 * Function to add two matrices 
 *
 * @param *mat_A[in]: Matrix A   
 * @param *mat_A[in]: Matrix B  
 * @param *result[out]: result   
 * @param rows[in]: number of rows of matrices
 * @param cols[in]: number of columns of matrices 
 *  
 */
void matAdd(float *mat_A, float *mat_B, float *result, int rows, int cols){

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result[i*cols+j] = mat_A[i*cols+j]+mat_B[i*cols+j];
        }
    }

}

/**
 * @brief matSub 
 *
 * Function to subtract two matrices 
 *
 * @param *mat_A[in]: Matrix A   
 * @param *mat_A[in]: Matrix B  
 * @param *result[out]: result   
 * @param rows[in]: number of rows of matrices
 * @param cols[in]: number of columns of matrices 
 *  
 */ 
void matSub(float *mat_A, float *mat_B, float *result, int rows, int cols){

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result[i*cols+j] = mat_A[i*cols+j]-mat_B[i*cols+j];
        }
    }

}

/**
 * @brief matTranspose 
 *
 * Function to transpose a Matrix 
 *
 * @param *mat_A[in]: Matrix to transpose  
 * @param *result[out]: result   
 * @param rows[in]: number of rows of matrices
 * @param cols[in]: number of columns of matrices 
 *  
 */ 
void matTranspose(float *mat_A, float *result, int rows, int cols){

	for(int i = 0; i < rows; ++i){
   		for(int j = 0; j < cols; ++j) {
      		result[j*rows+i] = mat_A[i*cols+j];
   		}
   	}

}

/**
 * @brief matInverse 
 *
 * Function to calculate the inverse of a 4x4 Matrix 
 *
 * @param *mat[in]: Matrix to transpose  
 * @param *inv[out]: result   
 *  
 */
void matInverse(float *mat, float *inv) {
    int i, j, k, row;
    float temp, pivot, multiplier;
    float identity[16] = {1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f};

    //copy the input matrix 
    for (i = 0; i < 16; i++) {
        inv[i] = mat[i];
    }

    //gaussian elimination
    for (i = 0; i < 4; i++) {
        pivot = inv[i*4+i];
        row = i;
        //find pivot element
        for (j = i+1; j < 4; j++) {
            if (fabs(inv[j*4+i]) > fabs(pivot)) {
                pivot = inv[j*4+i];
                row = j;
            }
        }

        if (row != i) {
            //swap rows 
            for (j = 0; j < 4; j++) {
                temp = inv[i*4+j];
                inv[i*4+j] = inv[row*4+j];
                inv[row*4+j] = temp;

                temp = identity[i*4+j];
                identity[i*4+j] = identity[row*4+j];
                identity[row*4+j] = temp;
            }
        }

        //divide pivot row by the pivot element
        for (j = 0; j < 4; j++) {
            inv[i*4+j] /= pivot;
            identity[i*4+j] /= pivot;
        }

        //subtract the pivot row from the other rows
        for (j = 0; j < 4; j++) {
            if (j != i) {
                multiplier = inv[j*4+i];

                for (k = 0; k < 4; k++) {
                    inv[j*4+k] -= multiplier * inv[i*4+k];
                    identity[j*4+k] -= multiplier * identity[i*4+k];
                }
            }
        }
    }

    //copy back the inverted matrix 
    for (i = 0; i < 16; i++) {
        inv[i] = identity[i];
    }
}
