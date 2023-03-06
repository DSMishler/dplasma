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
dplasma_zgemm_z_New_mtest(dplasma_enum_t transA, dplasma_enum_t transB,
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

static parsec_taskpool_t *
dplasma_zgemm_z_New_default(dplasma_enum_t transA, dplasma_enum_t transB,
                          dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                          dplasma_complex64_t beta,  parsec_tiled_matrix_t* C,
                          dplasma_info_t opt)
{
    parsec_taskpool_t* zgemm_tp;

    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
    dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            parsec_zgemm_NN_taskpool_t* tp;
            tp = parsec_zgemm_NN_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            parsec_zgemm_NT_taskpool_t* tp;
            tp = parsec_zgemm_NT_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            parsec_zgemm_TN_taskpool_t* tp;
            tp = parsec_zgemm_TN_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
        else {
            parsec_zgemm_TT_taskpool_t* tp;
            tp = parsec_zgemm_TT_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
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

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_z_New - Generates the taskpool that performs one of the following
 *  matrix-matrix operations. WARNING: The computations are not done by this call.
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zgemm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_z
 * @sa dplasma_zgemm_z_Destruct
 * @sa dplasma_cgemm_z_New
 * @sa dplasma_dgemm_z_New
 * @sa dplasma_sgemm_z_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgemm_z_New_ex( dplasma_enum_t transA, dplasma_enum_t transB,
                      dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                      dplasma_complex64_t beta,  parsec_tiled_matrix_t* C, dplasma_info_t opt)
{
    parsec_taskpool_t* zgemm_tp = NULL;

    /* Check input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_z_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_z_New", "illegal value of transB");
        return NULL /*-2*/;
    }

    if ( C->dtype & parsec_matrix_block_cyclic_type ) {
        zgemm_tp = dplasma_zgemm_z_New_mtest(transA, transB, alpha, A, B, beta, C, opt);
        return zgemm_tp;
    }
    zgemm_tp = dplasma_zgemm_z_New_default(transA, transB, alpha, A, B, beta, C, opt);
    return zgemm_tp;
}

parsec_taskpool_t*
dplasma_zgemm_z_New( dplasma_enum_t transA, dplasma_enum_t transB,
                   dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                   dplasma_complex64_t beta,  parsec_tiled_matrix_t* C)
{
    parsec_taskpool_t *tp;
    dplasma_info_t opt;
    dplasma_info_create(&opt);
    tp = dplasma_zgemm_z_New_ex(transA, transB, alpha, A, B, beta, C, opt);
    dplasma_info_free(&opt);
    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_z_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zgemm_z_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_z_New
 * @sa dplasma_zgemm_z
 *
 ******************************************************************************/
void
dplasma_zgemm_z_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgemm_NN_taskpool_t *zgemm_tp = (parsec_zgemm_NN_taskpool_t *)tp;
    dplasma_data_collection_t *ddc_A = NULL, *ddc_B = NULL, *ddc_C = NULL;

    if( zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_NN_MTEST ||
        zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_NT_MTEST ||
        zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_TN_MTEST ||
        zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_TT_MTEST) {
        parsec_zgemm_NN_mtest_taskpool_t *zgemm_mtest_tp = (parsec_zgemm_NN_mtest_taskpool_t *)tp;
        parsec_tiled_matrix_t* Cdist = (parsec_tiled_matrix_t*)zgemm_mtest_tp->_g_Cdist;
        if ( NULL != Cdist ) {
            parsec_tiled_matrix_destroy( Cdist );
            free( Cdist );
        }
        dplasma_clean_adtt_all_loc(zgemm_mtest_tp->_g_ddescA, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(zgemm_mtest_tp->_g_ddescB, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(zgemm_mtest_tp->_g_ddescC, MAX_SHAPES);

        ddc_A = zgemm_mtest_tp->_g_ddescA;
        ddc_B = zgemm_mtest_tp->_g_ddescB;
        ddc_C = zgemm_mtest_tp->_g_ddescC;
    } else if( zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_NN ||
               zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_NT ||
               zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_TN ||
               zgemm_tp->_g_gemm_type == DPLASMA_ZGEMM_TT) {
        dplasma_clean_adtt_all_loc(zgemm_tp->_g_ddescA, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(zgemm_tp->_g_ddescB, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(zgemm_tp->_g_ddescC, MAX_SHAPES);

        ddc_A = zgemm_tp->_g_ddescA;
        ddc_B = zgemm_tp->_g_ddescB;
        ddc_C = zgemm_tp->_g_ddescC;
    }

    parsec_taskpool_free(tp);

    /* free the dplasma_data_collection_t, after the tp stops referring to them */
    if(NULL != ddc_A)
        dplasma_unwrap_data_collection(ddc_A);
    if(NULL != ddc_B)
        dplasma_unwrap_data_collection(ddc_B);
    if(NULL != ddc_C)
        dplasma_unwrap_data_collection(ddc_C);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_z_New
 * @sa dplasma_zgemm_z_Destruct
 * @sa dplasma_cgemm_z
 * @sa dplasma_dgemm_z
 * @sa dplasma_sgemm_z
 *
 ******************************************************************************/
int
dplasma_zgemm_z( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               dplasma_complex64_t alpha, const parsec_tiled_matrix_t *A,
                                        const parsec_tiled_matrix_t *B,
               dplasma_complex64_t beta,        parsec_tiled_matrix_t *C)
{
    parsec_taskpool_t *parsec_zgemm = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    /* Check input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_z", "illegal value of transA");
        return -1;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_z", "illegal value of transB");
        return -2;
    }

    if ( transA == dplasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    } else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    if ( transB == dplasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    } else {
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_zgemm_z", "tile sizes have to match");
        return -101;
    }
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_zgemm_z", "sizes of matrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_zgemm_z", "start indexes have to match");
        return -101;
    }

    M = C->m;
    N = C->n;
    K = An;

    /* Quick return */
    if (M == 0 || N == 0 ||
        ((alpha == (dplasma_complex64_t)0.0 || K == 0) && beta == (dplasma_complex64_t)1.0))
        return 0;

    parsec_zgemm = dplasma_zgemm_z_New(transA, transB,
                                    alpha, A, B,
                                    beta, C);

    if ( parsec_zgemm != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_zgemm);
        dplasma_wait_until_completion(parsec);
        dplasma_zgemm_Destruct( parsec_zgemm );
        return 0;
    }
    return -101;
}
