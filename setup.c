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
}

/**
 * @brief Allocate all of the arrays used for computation
 * 
 */
void allocate_arrays(int recv_count, int col_length) {
	Ex_size_x = X; Ex_size_y = Y+1;
	Ex = alloc_2d_array(X, Y+1);
	Ey_size_x = X+1; Ey_size_y = Y;
	Ey = alloc_2d_array(X+1, Y);
	
	Bz_size_x = X; Bz_size_y = Y;
	Bz = alloc_2d_array(X, Y);
	
	E_size_x = X+1; E_size_y = Y+1; E_size_z = 3;
	E = alloc_3d_array(E_size_x, E_size_y, E_size_z);

	B_size_x = X+1; B_size_y = Y+1; B_size_z = 3;
	B = alloc_3d_array(B_size_x, B_size_y, B_size_z);

	// Create local matrix for alterations
	bz_local = alloc_2d_array(recv_count, col_length);
    ex_local = alloc_2d_array(recv_count, col_length+1);
    ey_local = alloc_2d_array(recv_count+1, col_length);
	bz_top_local = alloc_2d_array(1, col_length);
	local_buf = alloc_2d_array(1, col_length);
}

/**
 * @brief Free all of the arrays used for the computation
 * 
 */
void free_arrays() {
	free_2d_array(Ex);
	free_2d_array(Ey);
	free_2d_array(Bz);
	free_3d_array(E);
	free_3d_array(B);
	// MPI
	free_2d_array(bz_local);
	free_2d_array(ex_local);
	free_2d_array(ey_local);
}

/**
 * @brief Set up a guassian to curve around the centre
 * 
 */
void problem_set_up(int rank, int recv_count, int displacement, int col_length) {
    for (int i = 0; i < Ex_size_x; i++ ) {
        for (int j = 0; j < Ex_size_y; j++) {
            double xcen = lengthX / 2.0;
            double ycen = lengthY / 2.0;
            double xcoord = (i - xcen) * dx;
            double ycoord = j * dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
			double tx = (rlen == 0) ? 0 : ry / rlen;
            double mag = exp(-400.0 * (rlen - (lengthX / 4.0)) * (rlen - (lengthX / 4.0)));
            Ex[i][j] = mag * tx;
		}
	}
    for (int i = 0; i < Ey_size_x; i++ ) {
        for (int j = 0; j < Ey_size_y; j++) {
            double xcen = lengthX / 2.0;
            double ycen = lengthY / 2.0;
            double xcoord = i * dx;
            double ycoord = (j - ycen) * dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
            double ty = (rlen == 0) ? 0 : -rx / rlen;
			double mag = exp(-400.0 * (rlen - (lengthY / 4.0)) * (rlen - (lengthY / 4.0)));
            Ey[i][j] = mag * ty;
		}
	}

	for (int i=0; i<recv_count; i++) {
		int rel_i = displacement+i;
		for (int j=0; j<col_length; j++) {
			bz_local[i][j] = Bz[rel_i][j];
			ex_local[i][j] = Ex[rel_i][j];
			ey_local[i][j] = Ey[rel_i][j];
		}
		ex_local[i][col_length] = Ex[rel_i][col_length];
	}
	int rel_row = displacement+recv_count;
	for (int i=0; i<col_length; i++) {
		ey_local[recv_count][i] = Ey[rel_row][i];
	}
	//printf("%d: BEFORE\n", rank);
	// if (rank != 0) {
	// 	int rel_row = displacement-1;
	// 	for (int j=0; j<col_length; j++) {
	// 		bz_top_local[0][j] = Bz[rel_row][j];
	// 	}
	// }
	//printf("%d: AFTER\n", rank);
}
