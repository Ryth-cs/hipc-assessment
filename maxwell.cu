#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

// CUDA Version
#include <time.h>
#include <cstdlib>
#include <string.h>

/*
* Look for lines in the procfile contents like: 
* VmRSS:         5560 kB
* VmSize:         5560 kB
*
* Grab the number between the whitespace and the "kB"
* If 1 is returned in the end, there was a serious problem 
* (we could not find one of the memory usages)
*/
int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb)
{
    /* Get the the current process' status file from the proc filesystem */
    FILE* procfile = fopen("/proc/self/status", "r");

    long to_read = 8192;
    char buffer[to_read];
    int read = fread(buffer, sizeof(char), to_read, procfile);
    fclose(procfile);

    short found_vmrss = 0;
    short found_vmsize = 0;
    char* search_result;

    /* Look through proc status contents line by line */
    char delims[] = "\n";
    char* line = strtok(buffer, delims);

    while (line != NULL && (found_vmrss == 0 || found_vmsize == 0) )
    {
        search_result = strstr(line, "VmRSS:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmrss_kb);
            found_vmrss = 1;
        }

        search_result = strstr(line, "VmSize:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmsize_kb);
            found_vmsize = 1;
        }

        line = strtok(NULL, delims);
    }

    return (found_vmrss == 1 && found_vmsize == 1) ? 0 : 1;
}

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
__global__ void update_fields_gpu(double *Bz, double *Ex, double *Ey, int Bz_size_x, int Bz_size_y, int Ex_size_x, int Ex_size_y, int Ey_size_x, int Ey_size_y, double dx, double dy, double dt, double eps, double mu) {
	// Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < Bz_size_x && col < Bz_size_y) {
		// Legal move
		Bz[(row*Bz_size_y)+col] = Bz[(row*Bz_size_y)+col] - (dt / dx) * (Ey[((row+1)*Ey_size_y)+col] - Ey[(row*Ey_size_y)+col])
					  + (dt / dy) * (Ex[(row*Ex_size_y)+(col+1)] - Ex[(row*Ex_size_y)+col]);

		if (col != 0) {
			//Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
			Ex[(row*Ex_size_y)+col] = Ex[(row*Ex_size_y)+col] + (dt / (dy * eps * mu)) * (Bz[(row*Bz_size_y)+col] - Bz[(row*Bz_size_y)+(col-1)]);
		}

		if (row != 0) {
			//Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i][j] - Bz[i-1][j]);
			Ey[(row*Ey_size_y)+col] = Ey[(row*Ey_size_y)+col] - (dt / (dx * eps * mu)) * (Bz[(row*Bz_size_y)+col] - Bz[((row-1)*Bz_size_y)+col]);
		}
	}
}

/**
 * @brief Apply boundary conditions
 * 
 */
__global__ void apply_boundary_gpu(double *Ex, double *Ey, int Bz_size_x, int Bz_size_y, int Ex_size_x, int Ex_size_y, int Ey_size_x, int Ey_size_y) {
	// Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < Bz_size_x) {
		if (col == 0) {
			Ex[(row*Ex_size_y)] = -Ex[(row*Ex_size_y)+1];
		} else if (col == Ex_size_y-1) {
			Ex[(row*Ex_size_y)+col] = -Ex[(row*Ex_size_y)+(col-1)];
		}
	}
	
	if (col < Bz_size_y) {
		if (row == 0){
			//Ey[0][j] = -Ey[1][j];
			Ey[col] = -Ey[Ey_size_y+col];
		} else if (row == Ey_size_x-1){
			//Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
			Ey[(row*Ey_size_y)+col] = -Ey[((row-1)*Ey_size_y)+col];
		}
	}
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
void resolve_to_grid(double *E_mag, double *B_mag) {
	*E_mag = 0.0;
	*B_mag = 0.0;

	for (int k = 0; k < E_size_x*E_size_y*E_size_z; k+=E_size_z) {
		//int zDirection = k % E_size_z;
        int col = (k / E_size_z) % E_size_y;
        int row = k / (E_size_y * E_size_z);

		bool rowCheck = (row != 0) && (row != E_size_x-1);
		bool colCheck = (col != 0) && (col != E_size_y-1);
		if (rowCheck && colCheck) {
			//printf("X: %d, Y: %d, Z: %d\n", row, col, zDirection);

			E[k] = (Ex[((row-1)*Ex_size_y)+col] + Ex[(row*Ex_size_y)+col]) / 2.0;
			E[k+1] = (Ey[(row*Ey_size_y)+(col-1)] + Ey[(row*Ey_size_y)+col]) / 2.0;

			*E_mag += sqrt((E[k] * E[k]) + (E[k+1] * E[k+1]));
		}
	}
	
	for (int k = 0; k < B_size_x*B_size_y*B_size_z; k+=B_size_z) {
		//int zDirection = k % B_size_z;
        int col = (k / B_size_z) % B_size_y;
        int row = k / (B_size_y * B_size_z);

		bool rowCheck = (row != 0) && (row != B_size_x-1);
		bool colCheck = (col != 0) && (col != B_size_y-1);
		if (rowCheck && colCheck) {
			//printf("X: %d, Y: %d, Z: %d\n", row, col, zDirection);

			B[k+2] = (Bz[((row-1)*Bz_size_y)+col] + Bz[(row*Bz_size_y)+col] + Bz[(row*Bz_size_y)+(col-1)] + Bz[((row-1)*Bz_size_y)+(col-1)]) / 4.0;

			*B_mag += sqrt(B[k+2] * B[k+2]);
		}
	}
}

void printStats()
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("free: %lu\ntotal: %lu\n", freeMem, totalMem);
}

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	// CUDA timing setup
	printStats();
	long vmrss, vmsize;
	float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	set_defaults();
	parse_args(argc, argv);
	setup();

	printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X, Y);
	
	// Allocate arrays
	allocate_arrays();

	// Threads per CTA dimension
	int THREADS = 32;
	// Blocks per grid dimension
	int BLOCKS = (Bz_size_x + THREADS -1) / THREADS;

	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	//problem_set_up();
	problem_set_up_gpu<<<blocks, threads>>>(Ex, Ey, Bz_size_x, Bz_size_y, lengthX, lengthY, dx, dy);
	//cudaDeviceSynchronize();

	printf("AT START COMPLETED\n");

	// start at time 0
	double t = 0.0;
	int i = 0;
	while (i < steps) {
		//apply_boundary();
		apply_boundary_gpu<<<blocks, threads>>>(Ex, Ey, Bz_size_x, Bz_size_y, Ex_size_x, Ex_size_y, Ey_size_x, Ey_size_y);
		//update_fields();
		update_fields_gpu<<<blocks, threads>>>(Bz, Ex, Ey, Bz_size_x, Bz_size_y, Ex_size_x, Ex_size_y, Ey_size_x, Ey_size_y, dx, dy, dt, eps, mu);
		cudaDeviceSynchronize();

		t += dt;

		i++;
	}

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag);

	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
	printf("Simulation complete.\n");
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.1f ms \n", time);

	printStats();
	get_memory_usage_kb(&vmrss, &vmsize);
	printf("Current memory usage: VmRSS = %6ld KB %ld MB, VmSize = %6ld KB %ld MB\n", vmrss, (vmrss/1024), vmsize, (vmsize/1024));

	free_arrays();

	exit(0);
}


