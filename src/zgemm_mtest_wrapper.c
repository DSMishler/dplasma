/*
 * Copyright (c) 2010-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/types.h"
#include "dplasma/types_lapack.h"
#include "dplasmaaux.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#if defined(DPLASMA_HAVE_CUDA)
#include "parsec/mca/device/cuda/device_cuda.h"
#endif
#include "utils/dplasma_info.h"

#include "zgemm_NN.h"
#include "zgemm_NT.h"
#include "zgemm_TN.h"
#include "zgemm_TT.h"

#include "zgemm_NN_mtest.h"
#include "zgemm_NT_mtest.h"
#include "zgemm_TN_mtest.h"
#include "zgemm_TT_mtest.h"

#define MAX_SHAPES 3

/* no GPU for now */
/* #include "zgemm_NN_gpu.h" */

#include "parsec/utils/mca_param.h"

static parsec_taskpool_t *
dplasma_Zgemm_New_mtest(dplasma_enum_t transA, dplasma_enum_t transB,
                        dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                        dplasma_complex64_t beta,  parsec_tiled_matrix_t* C,
                        dplasma_info_t opt)
{
    int P, Q, IP, JQ, m, n;
    parsec_taskpool_t *zgemm_tp;
    parsec_matrix_block_cyclic_t *Cdist;

    P = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
    Q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;
    IP = ((parsec_matrix_block_cyclic_t*)C)->grid.ip;
    JQ = ((parsec_matrix_block_cyclic_t*)C)->grid.jq;

    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
    dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

    m = dplasma_imax(C->mt, P);
    n = dplasma_imax(C->nt, Q);

    /* Create a copy of the C matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Cdist = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    parsec_matrix_block_cyclic_init(
            Cdist, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            C->super.myrank,
            1, 1, /* Dimensions of the tiles              */
            m, n, /* Dimensions of the matrix             */
            0, 0, /* Starting points (not important here) */
            m, n, /* Dimensions of the submatrix          */
            P, Q, 1, 1, IP, JQ);
    Cdist->super.super.data_of = NULL;
    Cdist->super.super.data_of_key = NULL;

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NN_mtest\n");
            parsec_zgemm_NN_mtest_taskpool_t* tp;
            tp = parsec_zgemm_NN_mtest_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
            zgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NT_mtest\n");
            parsec_zgemm_NT_mtest_taskpool_t* tp;
            tp = parsec_zgemm_NT_mtest_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_TN_mtest\n");
            parsec_zgemm_TN_mtest_taskpool_t* tp;
            tp = parsec_zgemm_TN_mtest_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
            zgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_TT_mtest\n");
            parsec_zgemm_TT_mtest_taskpool_t* tp;
            tp = parsec_zgemm_TT_mtest_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C,
                                           (parsec_data_collection_t*)Cdist);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    }

    int shape = 0;
    dplasma_setup_adtt_all_loc( ddc_A,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    dplasma_setup_adtt_all_loc( ddc_B,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    dplasma_setup_adtt_all_loc( ddc_C,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    assert(shape == MAX_SHAPES);

    (void)opt; //No user-defined options for this algorithm
    return zgemm_tp;
}
