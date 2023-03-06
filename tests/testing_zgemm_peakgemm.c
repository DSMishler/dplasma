/*
 * Copyright (c) 2015-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 */



/* 
 * @precisions normal z -> s d c
 */
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#if !defined(DISABLE_SCHED_AFFINITY)
#include <linux/unistd.h>
#define gettid() syscall(__NR_gettid)
#include <sched.h>
#else
#warning "Not using sched affinity to set bindings!"
#endif /*!defined(DISABLE_SCHED_AFFINITY)*/

#include <cblas.h>


#include "common.h"
#include "dplasma/types.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function.h"

#define MAX_THREADS 128

static int NB = 120;
static volatile int running = 1;

// #define TYPE      dplasma_complex64_t
#define TYPE      double
#define GEMM_FUNC cblas_dgemm
/*
#define TYPE      double
#define GEMM_FUNC cblas_dgemm
*/
static void run_gemm(const TYPE *A, const TYPE *B, TYPE *C)
{
    GEMM_FUNC( CblasColMajor, 
               CblasNoTrans, CblasNoTrans,
               NB /* A.rows */, 
               NB /* A.cols */,
               NB /* B.rows */,
               1.0, (TYPE*)A, NB /* A.data_stride */,
                    (TYPE*)B, NB /* B.data_stride */,
               1.0, (TYPE*)C, NB /* C.data_stride */);
}
// static void run_gemm(const TYPE *A, const TYPE *B, TYPE *C)
// {
//     GEMM_FUNC( (const enum CBLAS_ORDER)CblasColMajor, 
//                (const enum CBLAS_TRANSPOSE)CblasNoTrans, (const enum CBLAS_TRANSPOSE)CblasNoTrans,
//                NB /* A.rows */, 
//                NB /* A.cols */,
//                NB /* B.rows */,
//                1.0, (TYPE*)A, NB /* A.data_stride */,
//                     (TYPE*)B, NB /* B.data_stride */,
//                1.0, (TYPE*)C, NB /* C.data_stride */);
// }

static TYPE *init_matrix(void)
{
    TYPE *res;
    int i, j;   res = (TYPE*)calloc(NB*NB, sizeof(TYPE));
    for(i = 0; i < NB; i++)
        for(j = 0; j < NB; j++)
            res[i*NB+j] = (TYPE)rand() / (TYPE)RAND_MAX;
    return res;
}

void perform_loop(void)
{
    TYPE *A, *B, *C;
    int i;
    unsigned long long int time;
    struct timespec start, end;
    int max_i = 100000000/NB/NB;
    if(max_i > 1024)
    {
       max_i = 1024;
    }
    else if (max_i < 5)
    {
       max_i = 5;
    }

    A = init_matrix();
    B = init_matrix();
    C = init_matrix();

    // for(i = 0; i <= min(max_i/5, 0); i++) {
        // /* Ensures that all threads run cblas_dgemm on another core */
        // /* while warming up */
        // run_gemm(A, B, C);
    // }

    for(i = 0; i < max_i; i++) {
        /* Take the time */
        clock_gettime(CLOCK_REALTIME, &start);
        run_gemm(A, B, C);
        clock_gettime(CLOCK_REALTIME, &end);
        time = end.tv_nsec - start.tv_nsec + ( (end.tv_sec - start.tv_sec) * 1000000000ULL );
        printf("NB = %d TIME = %llu ns  %f GFlops\n", NB, time,
                (2*(NB/1e3)*(NB/1e3)*(NB/1e3)) / ((double)time  / 1e9));
    }

    return;
}

int main(int argc, char *argv[])
{
    unsigned long int i, nbcores;
    pthread_t threads[MAX_THREADS];

    // this file will ignore extra arguments to match up with a python script
    if( argc < 3 ) {
        fprintf(stderr, "usage: %s -N <Matrix Size>\n", argv[0]);
        fprintf(stderr, "   also: I ignore extra arguments\n");
        return 1;
    }

    NB = atoi(argv[2]);

    if( NB <= 1 ) {
        fprintf(stderr, "usage: %s <Matrix Size>\n", argv[0]);
        return 1;
    }

    perform_loop();

    return 0;
}
