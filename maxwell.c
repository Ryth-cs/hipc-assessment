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

void printMatrix(double **values, int row, int col) {
    int i, j;

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            printf("%.02f ", values[i][j]);
        }
        printf("\n");
    }
}

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields(int rank, int size, int recv_count, int col_length, int step) {
	// 4000 x 4000
	// for (int i = 0; i < Bz_size_x; i++) {
	// 	for (int j = 0; j < Bz_size_y; j++) {
	// 		Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i+1][j] - Ey[i][j])
	// 			                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
	// 	}
	// }
	//printf("%d: Starting Bz step %d\n", rank, step);
	for (int i = 0; i < recv_count; i++) {
		for (int j = 0; j < col_length; j++) {
			bz_local[i][j] = bz_local[i][j] - (dt / dx) * (ey_local[i+1][j] - ey_local[i][j])
				                + (dt / dy) * (ex_local[i][j+1] - ex_local[i][j]);
		}
	}
	//printf("%d: Finished Bz step %d\n", rank, step);
	// if (rank != 0) {
	// 	for (int j = 0; j < col_length; j++) {
	// 		bz_top_local[0][j] = bz_top_local[0][j] - (dt / dx) * (ey_local[i+1][j] - ey_local[i][j])
	// 			                + (dt / dy) * (ex_local[i][j+1] - ex_local[i][j]);
	// 	}
	// }
	// Send bottom of Bz to bz top row
	int send_rank = (rank == size-1) ? MPI_PROC_NULL : rank+1;
    int recv_rank = (rank == 0) ? MPI_PROC_NULL : rank-1;
	// printf("%d: Bz matrix step %d\n", rank, step);
	// printMatrix(bz_local, recv_count, col_length);
	// printf("\n");

	// printf("%d: Bz local send step %d\n", rank, step);
	// for (int j=0; j<col_length; j++) {
	// 	printf("%.2f ", bz_local[recv_count-1][j]);
	// }
	// printf("\n");

	// printf("%d: Bz local recv step %d\n", rank, step);
	// for (int j=0; j<col_length; j++) {
	// 	printf("%.2f ", bz_top_local[0][j]);
	// }
	// printf("\n");

	// printf("%d: send to %d, recv from %d\n", rank, send_rank, recv_rank);
	// printf("%d: recv count %d step %d\n", rank, recv_count, step);
	MPI_Sendrecv(&bz_local[recv_count-1][0], 1, bzType, send_rank, 0, &bz_top_local[0][0], 1, bzType, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// printf("%d: Successfully sent BZ step %d\n", rank, step);

	// printf("%d: New Bz local top step %d\n", rank, step);
	// for (int j=0; j<col_length; j++) {
	// 	printf("%.2f ", bz_top_local[0][j]);
	// }
	// printf("\n");

	// 4000 x 4000
	// for (int i = 0; i < Ex_size_x; i++) {
	// 	for (int j = 1; j < Ex_size_y-1; j++) {
	// 		Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
	// 	}
	// }
	// printf("%d: Starting Ex\n", rank);
	for (int i = 0; i < recv_count; i++) {
		for (int j = 1; j < col_length; j++) {
			ex_local[i][j] = ex_local[i][j] + (dt / (dy * eps * mu)) * (bz_local[i][j] - bz_local[i][j-1]);
		}
	}

	// 4000 x 4000
	// for (int i = 1; i < Ey_size_x-1; i++) {
	// 	for (int j = 0; j < Ey_size_y; j++) {
	// 		Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i][j] - Bz[i-1][j]);
	// 	}
	// }
	//int start_row = (rank == 0) ? 1 : 0;
	int finish_row = (rank == size-1) ? recv_count : recv_count;
	//printf("%d: start: %d, finish: %d\n", rank, 1, finish_row);
	for (int i = 1; i < finish_row; i++) {
		for (int j = 0; j < col_length; j++) {
			ey_local[i][j] = ey_local[i][j] - (dt / (dx * eps * mu)) * (bz_local[i][j] - bz_local[i-1][j]); // Trying to reference outside area
		}
	}
	if (rank != 0) {
		for (int j=0; j<col_length; j++) {
			ey_local[0][j] = ey_local[0][j] - (dt / (dx * eps * mu)) * (bz_local[0][j] - bz_top_local[0][j]);
			//printf("%d: ey local top - %f\n", rank, ey_local[0][j]);
		}
		
	}
	//printf("%d: EY EDIT SUCCESS\n", rank);
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary(int rank, int size, int recv_count) {
	// for (int i = 0; i < Ex_size_x; i++) {
	// 	Ex[i][0] = -Ex[i][1];
	// 	Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	// }
	for (int i=0; i<recv_count; i++) {
		ex_local[i][0] = -ex_local[i][1];
		ex_local[i][Ex_size_y-1] = -ex_local[i][Ex_size_y-2];
	}

	// for (int j = 0; j < Ey_size_y; j++) {
	// 	Ey[0][j] = -Ey[1][j];
	// 	Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	// }
	if (rank == 0) {
		for (int j = 0; j < Ey_size_y; j++) {
			ey_local[0][j] = -ey_local[1][j];
		}
	} else if (rank == size-1) {
		for (int j = 0; j < Ey_size_y; j++) {
			ey_local[recv_count][j] = -ey_local[recv_count-1][j];
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
	//printf("%d: Time taken to setup: %f\n", rank, MPI_Wtime()-start_time);

	//printf("%d: Running problem size %f x %f on a %d x %d grid.\n", rank, lengthX, lengthY, X, Y);

	if (verbose) print_opts();

	int row_length = X;
	int col_length = Y;
	// Create datatype for Bz
    MPI_Type_vector(1, col_length, 0, MPI_DOUBLE, &bzType);
    MPI_Type_commit(&bzType);
	// Create datatype for Ex
    MPI_Type_vector(1, col_length+1, 0, MPI_DOUBLE, &exType);
    MPI_Type_commit(&exType);
	// Create datatype for Ey
    MPI_Type_vector(1, col_length, 0, MPI_DOUBLE, &eyType);
    MPI_Type_commit(&eyType);

	// Calculate the no. of rows for each process and calculate offset
    int interval, modulus;
    int recvCounts[size]; // Number of rows to be received
    int displs[size]; // Displacement offset from 0 and starting value

	// Bz calcs
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
	// for (int i=0; i<size; i++) {
	// 	printf("Rank: %d, rec: %d, disp: %d\n", i, recvCounts[i], displs[i]);
	// }

	allocate_arrays(recvCounts[rank], col_length);

	problem_set_up(rank, recvCounts[rank], displs[rank], col_length);

	// printf("%d: BZ TOP\n", rank);
	// printMatrix(bz_top_local, 1, col_length);

	//printf("%d: Time taken to get to start: %f\n", rank, MPI_Wtime()-start_time);


	// printf("%d: ey local\n", rank);
	// printMatrix(ey_local, recvCounts[rank]+1, col_length);
	// printf("\n");


	double t = 0.0;
	// start at time 0
	int i = 0;
	while(i < steps) {
		apply_boundary(rank, size, recvCounts[rank]);

		// printf("%d: Ey post boundary step: %d\n", rank, i);
		// printMatrix(ey_local, recvCounts[rank]+1, col_length);
		// printf("\n");

		update_fields(rank, size, recvCounts[rank], col_length, i);

		t += dt;

		// printf("%d: Ey before recv\n", rank);
		// printMatrix(ey_local, recvCounts[rank]+1, col_length);
		// printf("\n");

		// Grab top row from next rank and replace bottom of current
        int send_rank = (rank == 0) ? MPI_PROC_NULL : rank-1;
        int recv_rank = (rank == size-1) ? MPI_PROC_NULL : rank+1;
		// Send my top row to rank-1 bottom row
        MPI_Sendrecv(&ey_local[0][0], 1, eyType, send_rank, 0, &ey_local[recvCounts[rank]][0], 1, eyType, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// // Send bottom of Bz to bz top row
		// send_rank = (rank == size-1) ? MPI_PROC_NULL : rank+1;
        // recv_rank = (rank == 0) ? MPI_PROC_NULL : rank-1;
		// MPI_Sendrecv(&bz_local[recvCounts[rank]-1][0], 1, bzType, send_rank, 0, &bz_top_local[0][0], 1, bzType, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// printf("%d: Ey post recv\n", rank);
		// printMatrix(ey_local, recvCounts[rank]+1, col_length);
		// printf("\n");

		// printf("%d: Ey step: %d\n", rank, i);
		// printMatrix(ey_local, recvCounts[rank]+1, col_length);
		// printf("\n");
		

		i++;
	}

	// Gather all the data back together
	MPI_Allgatherv(&(bz_local[0][0]), recvCounts[rank], bzType,
			&(Bz[0][0]), recvCounts, displs, bzType,
			MPI_COMM_WORLD);
	MPI_Allgatherv(&(ex_local[0][0]), recvCounts[rank], exType,
			&(Ex[0][0]), recvCounts, displs, exType,
			MPI_COMM_WORLD);
	recvCounts[size-1] = recvCounts[size-1] + 1;
	MPI_Allgatherv(&(ey_local[0][0]), recvCounts[rank], eyType,
			&(Ey[0][0]), recvCounts, displs, eyType,
			MPI_COMM_WORLD);

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag);

	if (rank == 0) {
		printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", steps, t, dt, E_mag, B_mag);
		printf("Simulation complete.\n");
		printf("Time taken to compute: %f\n", MPI_Wtime()-start_time);
	}
	
	if (!no_output) 
		write_result();

	free_arrays();

	MPI_Finalize();
	exit(0);
}


