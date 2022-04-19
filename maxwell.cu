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

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields() {
	// for (int i = 0; i < Bz_size_x; i++) {
	// 	for (int j = 0; j < Bz_size_y; j++) {
	// 		Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i+1][j] - Ey[i][j])
	// 			                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
	// 	}
	// }
	for (int k = 0; k < Bz_size_x*Bz_size_y; k++) {
		int row = k / Bz_size_y;
		int col = k % Bz_size_y;
		Bz[k] = Bz[k] - (dt / dx) * (Ey[((row+1)*Ey_size_y)+col] - Ey[(row*Ey_size_y)+col])
					  + (dt / dy) * (Ex[(row*Ex_size_y)+(col+1)] - Ex[(row*Ex_size_y)+col]);
	}

	// for (int i = 0; i < Ex_size_x; i++) {
	// 	for (int j = 1; j < Ex_size_y-1; j++) {
	// 		Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
	// 	}
	// }
	for (int k = 0; k < Ex_size_x*Ex_size_y; k++) {
		int row = k / Ex_size_y;
		int col = k % Ex_size_y;
		if (col != 0 && col != (Ex_size_y-1)){
			Ex[k] = Ex[k] + (dt / (dy * eps * mu)) * (Bz[(row*Bz_size_y)+col] - Bz[(row*Bz_size_y)+(col-1)]);
		}
	}

	// for (int i = 1; i < Ey_size_x-1; i++) {
	// 	for (int j = 0; j < Ey_size_y; j++) {
	// 		Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i][j] - Bz[i-1][j]);
	// 	}
	// }
	for (int k = 0; k < Ey_size_x*Ey_size_y; k++) {
		int row = k / Ey_size_y;
		int col = k % Ey_size_y;
		if (row != 0 && row != (Ey_size_x-1)){
			Ey[k] = Ey[k] - (dt / (dx * eps * mu)) * (Bz[(row*Bz_size_y)+col] - Bz[((row-1)*Bz_size_y)+col]);
		}
	}
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary() {
	// for (int i = 0; i < Ex_size_x; i++) {
	// 	Ex[i][0] = -Ex[i][1];
	// 	Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	// }
	for (int k = 0; k < Ex_size_x*Ex_size_y; k+=Ex_size_y) {
        Ex[k] = -Ex[k+1];
		Ex[k+(Ex_size_y-1)] = -Ex[k+(Ex_size_y-2)];
    }

	// for (int j = 0; j < Ey_size_y; j++) {
	// 	Ey[0][j] = -Ey[1][j];
	// 	Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	// }
	for (int k = 0; k < Ey_size_y; k++) {
		Ey[k] = -Ey[k+Ey_size_y];
		Ey[k+(Ey_size_y*(Ey_size_x-1))] = -Ey[k+(Ey_size_y*(Ey_size_x-2))];
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

	// for (int i = 1; i < E_size_x-1; i++) {
	// 	for (int j = 1; j < E_size_y-1; j++) {
	// 		E[i][j][0] = (Ex[i-1][j] + Ex[i][j]) / 2.0;
	// 		E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;
	// 		//E[i][j][2] = 0.0; // in 2D we don't care about this dimension

	// 		*E_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
	// 	}
	// }
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
	
	// for (int i = 1; i < B_size_x-1; i++) {
	// 	for (int j = 1; j < B_size_y-1; j++) {
	// 		//B[i][j][0] = 0.0; // in 2D we don't care about these dimensions
	// 		//B[i][j][1] = 0.0;
	// 		B[i][j][2] = (Bz[i-1][j] + Bz[i][j] + Bz[i][j-1] + Bz[i-1][j-1]) / 4.0;

	// 		*B_mag += sqrt(B[i][j][2] * B[i][j][2]);
	// 	}
	// }
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

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	// CUDA timing setup
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

	problem_set_up();

	printf("AT START COMPLETED\n");

	// printf("Time taken to get to start: %f\n", (( double ) ( clock() - start ) / CLOCKS_PER_SEC));

	// start at time 0
	double t = 0.0;
	int i = 0;
	while (i < steps) {
		apply_boundary();
		update_fields();

		t += dt;

		// if (i % output_freq == 0) {
		// 	double E_mag, B_mag;
		// 	resolve_to_grid(&E_mag, &B_mag);
		// 	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);

		// 	if ((!no_output) && (enable_checkpoints))
		// 		write_checkpoint(i);
		// }

		i++;
	}

	printf("Bz Below\n");
	for (int k=0; k<Bz_size_x*Bz_size_y; k+=Bz_size_y) {
		for (int col=0; col<Bz_size_y; col++) {
			printf("%f ", Bz[k+col]);
		}
		printf("\n");
	}

	printf("Ex Below\n");
	for (int k=0; k<Ex_size_x*Ex_size_y; k+=Ex_size_y) {
		for (int col=0; col<Ex_size_y; col++) {
			printf("%f ", Ex[k+col]);
		}
		printf("\n");
	}

	printf("Ey Below\n");
	for (int k=0; k<Ey_size_x*Ey_size_y; k+=Ey_size_y) {
		for (int col=0; col<Ey_size_y; col++) {
			printf("%f ", Ey[k+col]);
		}
		printf("\n");
	}
	//exit(0);

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag);

	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
	printf("Simulation complete.\n");
	// //printf("Time taken to compute: %f\n", (( double ) ( clock() - start ) / CLOCKS_PER_SEC));
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.1f ms \n", time);

	// if (!no_output) 
	// 	write_result();

	free_arrays();

	exit(0);
}


