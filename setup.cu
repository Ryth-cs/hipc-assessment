#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Set up some default values before arguments have been loaded
 * 
 */
void set_defaults() {
	lengthX = 1.0;
	lengthY = 1.0;

	X = 4000;
	Y = 4000;
	
	T = 1.6e-9;

	set_default_base();
}

/**
 * @brief Set up some of the values required for computation after arguments have been loaded
 * 
 */
void setup() {
	dx = lengthX / X;
	dy = lengthY / Y;

	dt = cfl * (dx > dy ? dx : dy) / c;
	
	if (steps == 0) // only set this if steps hasn't been specified
		steps = (int) (T / dt);

	// // Pass to GPU
	// cudaMalloc((void**)&d_lengthX, sizeof(double));
	// cudaMemcpy(d_lengthX, &lengthX, sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Allocate all of the arrays used for computation
 * 
 */
void allocate_arrays() {
	// Ex_size_x = X; Ex_size_y = Y+1;
	// Ex = alloc_2d_array(X, Y+1);
	// Ey_size_x = X+1; Ey_size_y = Y;
	// Ey = alloc_2d_array(X+1, Y);
	// Bz_size_x = X; Bz_size_y = Y;
	// Bz = alloc_2d_array(X, Y);
	Bz_size_x = X; Bz_size_y = Y;
	Ex_size_x = X; Ex_size_y = Y+1;
	Ey_size_x = X+1; Ey_size_y = Y;
	
	size_t Bz_bytes = Bz_size_x * Bz_size_y * sizeof(double);
	size_t Ex_bytes = Ex_size_x * Ex_size_y * sizeof(double);
	size_t Ey_bytes = Ey_size_x * Ey_size_y * sizeof(double);

	cudaMallocManaged(&Bz, Bz_bytes);
	cudaMallocManaged(&Ex, Ex_bytes);
	cudaMallocManaged(&Ey, Ey_bytes);

	// E_size_x = X+1; E_size_y = Y+1; E_size_z = 3;
	// E = alloc_3d_array(E_size_x, E_size_y, E_size_z);

	// B_size_x = X+1; B_size_y = Y+1; B_size_z = 3;
	// B = alloc_3d_array(B_size_x, B_size_y, B_size_z);

	E_size_x = X+1; E_size_y = Y+1; E_size_z = 3;
	B_size_x = X+1; B_size_y = Y+1; B_size_z = 3;

	size_t E_bytes = E_size_x * E_size_y * E_size_z * sizeof(double);
	size_t B_bytes = B_size_x * B_size_y * B_size_z * sizeof(double);

	cudaMallocManaged(&E, E_bytes);
	cudaMallocManaged(&B, B_bytes);
}

/**
 * @brief Free all of the arrays used for the computation
 * 
 */
void free_arrays() {
	// free_2d_array(Ex);
	// free_2d_array(Ey);
	// free_2d_array(Bz);

	// Deallocate device memory
    cudaFree(Bz);
    cudaFree(Ex);
    cudaFree(Ey);

	// free_3d_array(E);
	// free_3d_array(B);
	cudaFree(E);
	cudaFree(B);
}

/**
 * @brief Set up a guassian to curve around the centre
 * 
 */
void problem_set_up() {
	for (int k=0; k<Ex_size_x*Ex_size_y; k++) {
		int row = k / Ex_size_y;
		int col = k % Ex_size_y;
		double xcen = lengthX / 2.0;
		double ycen = lengthY / 2.0;
		double xcoord = (row - xcen) * dx;
		double ycoord = col * dy;
		double rx = xcen - xcoord;
		double ry = ycen - ycoord;
		double rlen = sqrt(rx*rx + ry*ry);
		double tx = (rlen == 0) ? 0 : ry / rlen;
		double mag = exp(-400.0 * (rlen - (lengthX / 4.0)) * (rlen - (lengthX / 4.0)));
		//Ex[i][j] = mag * tx;
		Ex[k] = mag * tx;
	}

	for (int k=0; k<Ey_size_x*Ey_size_y; k++) {
		int row = k / Ey_size_y;
		int col = k % Ey_size_y;
		double xcen = lengthX / 2.0;
		double ycen = lengthY / 2.0;
		double xcoord = row * dx;
		double ycoord = (col - ycen) * dy;
		double rx = xcen - xcoord;
		double ry = ycen - ycoord;
		double rlen = sqrt(rx*rx + ry*ry);
		double ty = (rlen == 0) ? 0 : -rx / rlen;
		double mag = exp(-400.0 * (rlen - (lengthY / 4.0)) * (rlen - (lengthY / 4.0)));
		//Ey[i][j] = mag * ty;
		Ey[k] = mag*ty;
	}
}

__global__ void problem_set_up_gpu(double *Ex, double *Ey, int Bz_size_x, int Bz_size_y, double lengthX, double lengthY, double dx, double dy) {
	// Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Boundary check for matrix
	if (row < Bz_size_x){
		if (col < Bz_size_y){
			// Do Ex Ey
			double xcen = lengthX / 2.0;
			double ycen = lengthY / 2.0;
			double xcoord = (row - xcen) * dx;
			double ycoord = col * dy;
			double rx = xcen - xcoord;
			double ry = ycen - ycoord;
			double rlen = sqrt(rx*rx + ry*ry);
			double tx = (rlen == 0) ? 0 : ry / rlen;
			double mag = exp(-400.0 * (rlen - (lengthX / 4.0)) * (rlen - (lengthX / 4.0)));
			//Ex[i][j] = mag * tx;
			Ex[row * (Bz_size_y+1) + col] = mag * tx;

			xcen = lengthX / 2.0;
			ycen = lengthY / 2.0;
			xcoord = row * dx;
			ycoord = (col - ycen) * dy;
			rx = xcen - xcoord;
			ry = ycen - ycoord;
			rlen = sqrt(rx*rx + ry*ry);
			double ty = (rlen == 0) ? 0 : -rx / rlen;
			mag = exp(-400.0 * (rlen - (lengthY / 4.0)) * (rlen - (lengthY / 4.0)));
			//Ey[i][j] = mag * ty;
			Ey[row * Bz_size_y + col] = mag*ty;

		} else if (col == Bz_size_y) {
			// Do Ex extra
			double xcen = lengthX / 2.0;
			double ycen = lengthY / 2.0;
			double xcoord = (row - xcen) * dx;
			double ycoord = col * dy;
			double rx = xcen - xcoord;
			double ry = ycen - ycoord;
			double rlen = sqrt(rx*rx + ry*ry);
			double tx = (rlen == 0) ? 0 : ry / rlen;
			double mag = exp(-400.0 * (rlen - (lengthX / 4.0)) * (rlen - (lengthX / 4.0)));
			//Ex[i][j] = mag * tx;
			Ex[row * Bz_size_y + col] = mag * tx;
		
		}
	} else if (row == Bz_size_x) {
		if (col < Bz_size_y){
			// Do Ey extra
			double xcen = lengthX / 2.0;
			double ycen = lengthY / 2.0;
			double xcoord = row * dx;
			double ycoord = (col - ycen) * dy;
			double rx = xcen - xcoord;
			double ry = ycen - ycoord;
			double rlen = sqrt(rx*rx + ry*ry);
			double ty = (rlen == 0) ? 0 : -rx / rlen;
			double mag = exp(-400.0 * (rlen - (lengthY / 4.0)) * (rlen - (lengthY / 4.0)));
			//Ey[i][j] = mag * ty;
			Ey[row * Bz_size_y + col] = mag*ty;
		}
	}
}
