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
void update_fields(int rank, int size, int recv_count, int col_length, int step, MPI_Datatype bzType) {
	// 4000 x 4000
	// for (int i = 0; i < Bz_size_x; i++) {
	// 	for (int j = 0; j < Bz_size_y; j++) {
	// 		Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i+1][j] - Ey[i][j])
	// 			                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
	// 	}
	// }
	for (int i = 0; i < recv_count; i++) {
		for (int j = 0; j < col_length; j++) {
			bz_local[i][j] = bz_local[i][j] - (dt / dx) * (ey_local[i+1][j] - ey_local[i][j])
				                + (dt / dy) * (ex_local[i][j+1] - ex_local[i][j]);
		}
	}

	// Send bottom of Bz to bz top row
	int send_rank = (rank == size-1) ? MPI_PROC_NULL : rank+1;
    int recv_rank = (rank == 0) ? MPI_PROC_NULL : rank-1;
	MPI_Sendrecv(&bz_local[recv_count-1][0], 1, bzType, send_rank, 0, &bz_top_local[0][0], 1, bzType, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// 4000 x 4000
	// for (int i = 0; i < Ex_size_x; i++) {
	// 	for (int j = 1; j < Ex_size_y-1; j++) {
	// 		Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
	// 	}
	// }
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
	int finish_row = (rank == size-1) ? recv_count : recv_count;
	for (int i = 1; i < finish_row; i++) {
		for (int j = 0; j < col_length; j++) {
			ey_local[i][j] = ey_local[i][j] - (dt / (dx * eps * mu)) * (bz_local[i][j] - bz_local[i-1][j]); // Trying to reference outside area
		}
	}
	if (rank != 0) {
		for (int j=0; j<col_length; j++) {
			ey_local[0][j] = ey_local[0][j] - (dt / (dx * eps * mu)) * (bz_local[0][j] - bz_top_local[0][j]);
		}
		
	}
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
	long vmrss, vmsize;

	//printf("Version: MPI\n");
	set_defaults();
	parse_args(argc, argv);
	setup();

	if (rank == 0){
		printf("%d: Time taken to setup: %f\n", rank, MPI_Wtime()-start_time);
		printf("%d: Running problem size %f x %f on a %d x %d grid.\n", rank, lengthX, lengthY, X, Y);
	}

	if (verbose) print_opts();

	int row_length = X;
	int col_length = Y;
	// Create datatype for Bz
	MPI_Datatype bzType;
    MPI_Type_vector(1, col_length, 0, MPI_DOUBLE, &bzType);
    MPI_Type_commit(&bzType);
	// Create datatype for Ex
	MPI_Datatype exType;
    MPI_Type_vector(1, col_length+1, 0, MPI_DOUBLE, &exType);
    MPI_Type_commit(&exType);
	// Create datatype for Ey
	MPI_Datatype eyType;
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

	allocate_arrays(recvCounts[rank], col_length, rank);

	problem_set_up(rank, recvCounts[rank], displs[rank], col_length);

	double t = 0.0;
	// start at time 0
	int i = 0;
	while(i < steps) {
		apply_boundary(rank, size, recvCounts[rank]);

		update_fields(rank, size, recvCounts[rank], col_length, i, bzType);

		t += dt;

		// Grab top row from next rank and replace bottom of current
        int send_rank = (rank == 0) ? MPI_PROC_NULL : rank-1;
        int recv_rank = (rank == size-1) ? MPI_PROC_NULL : rank+1;
		// Send my top row to rank-1 bottom row
        MPI_Sendrecv(&ey_local[0][0], 1, eyType, send_rank, 0, &ey_local[recvCounts[rank]][0], 1, eyType, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		i++;
	}

	// Gather all the data back together
	if (rank == 0) {
		MPI_Gatherv(&(bz_local[0][0]), recvCounts[rank], bzType,
				&(Bz[0][0]), recvCounts, displs, bzType, 0,
				MPI_COMM_WORLD);
		MPI_Gatherv(&(ex_local[0][0]), recvCounts[rank], exType,
				&(Ex[0][0]), recvCounts, displs, exType, 0,
				MPI_COMM_WORLD);
		recvCounts[size-1] = recvCounts[size-1] + 1;
		MPI_Gatherv(&(ey_local[0][0]), recvCounts[rank], eyType,
				&(Ey[0][0]), recvCounts, displs, eyType, 0,
				MPI_COMM_WORLD);
	} else {
		MPI_Gatherv(&(bz_local[0][0]), recvCounts[rank], bzType,
				NULL, recvCounts, displs, bzType, 0,
				MPI_COMM_WORLD);
		MPI_Gatherv(&(ex_local[0][0]), recvCounts[rank], exType,
				NULL, recvCounts, displs, exType, 0,
				MPI_COMM_WORLD);
		recvCounts[size-1] = recvCounts[size-1] + 1;
		MPI_Gatherv(&(ey_local[0][0]), recvCounts[rank], eyType,
				NULL, recvCounts, displs, eyType, 0,
				MPI_COMM_WORLD);
	}

	if (rank == 0) {
		double E_mag, B_mag;
		resolve_to_grid(&E_mag, &B_mag);
		printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", steps, t, dt, E_mag, B_mag);
		printf("Simulation complete.\n");
		printf("Time taken to compute: %f\n", MPI_Wtime()-start_time);
	}

	get_memory_usage_kb(&vmrss, &vmsize);
	printf("%d: Current memory usage: VmRSS = %6ld KB %ld MB, VmSize = %6ld KB %ld MB\n", rank, vmrss, (vmrss/1024), vmsize, (vmsize/1024));

	free_arrays(rank);

	MPI_Finalize();
	exit(0);
}


