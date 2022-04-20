#ifndef SETUP_H
#define SETUP_H

void set_defaults();
void setup();
void allocate_arrays();
void free_arrays();
void problem_set_up();
__global__ void problem_set_up_gpu(double *Ex, double *Ey, int Bz_size_x, int Bz_size_y, double lengthX, double lengthY, double dx, double dy);

#endif