#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

// MPI Version
#include <time.h>
#include <mpi.h>

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields(int rank, int displs[], int recvCounts[], int col_length) {
	// 4000 x 4000
	for (int i=0; i<recvCounts[rank]; i++) {
		for (int j=0; j<col_length; j++) {
			//send[i][j] = recv[i][j]+num;
			//send[i][j] = rank;
			int rel_i = displs[rank]+i;
			send_array[i][j] = send_array[i][j] - (dt / dx) * (Ey[rel_i+1][j] - Ey[rel_i][j])
								+ (dt / dy) * (Ex[rel_i][j+1] - Ex[rel_i][j]);
		}
	}
	// Gather the vector to complete matrix on all processes
	MPI_Allgatherv(&(send_array[0][0]), recvCounts[rank], arrType,
			&(Bz[0][0]), recvCounts, displs, arrType,
			MPI_COMM_WORLD);


	// 4000 x 4000
	for (int i = 0; i < Ex_size_x; i++) {
		for (int j = 1; j < Ex_size_y-1; j++) {
			Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
		}
	}

	// 4000 x 4000
	for (int i = 1; i < Ey_size_x-1; i++) {
		for (int j = 0; j < Ey_size_y; j++) {
			Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i][j] - Bz[i-1][j]);
		}
	}
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary() {
	for (int i = 0; i < Ex_size_x; i++) {
		Ex[i][0] = -Ex[i][1];
		Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	}

	for (int j = 0; j < Ey_size_y; j++) {
		Ey[0][j] = -Ey[1][j];
		Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
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

	for (int i = 1; i < E_size_x-1; i++) {
		for (int j = 1; j < E_size_y-1; j++) {
			E[i][j][0] = (Ex[i-1][j] + Ex[i][j]) / 2.0;
			E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;
			//E[i][j][2] = 0.0; // in 2D we don't care about this dimension

			*E_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
		}
	}
	
	for (int i = 1; i < B_size_x-1; i++) {
		for (int j = 1; j < B_size_y-1; j++) {
			//B[i][j][0] = 0.0; // in 2D we don't care about these dimensions
			//B[i][j][1] = 0.0;
			B[i][j][2] = (Bz[i-1][j] + Bz[i][j] + Bz[i][j-1] + Bz[i-1][j-1]) / 4.0;

			*B_mag += sqrt(B[i][j][2] * B[i][j][2]);
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

	// MPI Setup
	int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Set initial timer
	double start_time = MPI_Wtime();

	//printf("Version: MPI\n");
	set_defaults();
	parse_args(argc, argv);
	setup();
	printf("%d: Time taken to setup: %f\n", rank, MPI_Wtime()-start_time);

	printf("%d: Running problem size %f x %f on a %d x %d grid.\n", rank, lengthX, lengthY, X, Y);

	if (verbose) print_opts();
	
	allocate_arrays();

	problem_set_up();

	int row_length = X;
	int col_length = Y;
	// Create datatype
    //MPI_Datatype arrType;
    MPI_Type_vector(1, col_length, 0, MPI_DOUBLE, &arrType);
    MPI_Type_commit(&arrType);

	// Calculate the no. of rows for each process and calculate offset
    int interval, modulus;
    int recvCounts[size]; // Number of rows to be received
    int displs[size]; // Displacement offset from 0 and starting value
    interval = row_length/size;
    modulus = row_length % size;
    for (int i=0; i < size; i++) {
        if (modulus != 0) {
            recvCounts[i] = interval+1;
            modulus--;
        } else {
            recvCounts[i] = interval;
        }
        displs[i] = (i == 0) ? 0 : displs[i-1]+recvCounts[i-1];
    }

	// Create local matrix for alterations
    send_array = alloc_2d_array(recvCounts[rank], col_length);

	printf("%d: Time taken to get to start: %f\n", rank, MPI_Wtime()-start_time);

	// MPI_Finalize();
	// exit(0);

	///////////////////////////////////////
	double t = 0.0;
	// start at time 0
	for (int i = 0; i < steps; i++) {
		apply_boundary();
		update_fields(rank, displs, recvCounts, col_length);

		t += dt;
		if (i % output_freq == 0) {
			double E_mag, B_mag;
			resolve_to_grid(&E_mag, &B_mag);
			printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);

			if ((!no_output) && (enable_checkpoints))
				write_checkpoint(i);
		}
	}

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag);

	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", steps, t, dt, E_mag, B_mag);
	printf("Simulation complete.\n");
	printf("Time taken to compute: %f\n", MPI_Wtime()-start_time);

	if (!no_output) 
		write_result();

	free_arrays();

	MPI_Finalize();
	exit(0);
}


