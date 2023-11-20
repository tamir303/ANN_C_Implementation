#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>

#define assert_n(expr, msg) \
 if (!(expr)) { \
	fprintf(stderr, "Assertion failed :%s\n", msg); \
	assert(expr); \
 }

typedef double* vector_t;

void* dynamicAllocation(size_t);
void free_2d_array(void** arr, size_t rows);

vector_t vector_init(size_t);
double vector_prod(vector_t v1, vector_t v2, int size);
vector_t vector_mul(vector_t v1, vector_t v2, int size);
void vector_free(vector_t v);

int argmax(vector_t v, int size);

#endif