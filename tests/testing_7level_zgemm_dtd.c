/*
 * Copyright (c) 2015-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
/* Daniel Mishler */
/* seven level zgemm using PaRSEC */
/*
 * using the dplasma zgemm as a template. There are few differences
 * between the two implementations as of 2022-01-31 
 */

#include "common.h"
#include "dplasma/types.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function.h"

/* Global index for the full tile datatype */
static int TILE_FULL;

static int check_solution( parsec_context_t *parsec, int loud,
                           dplasma_enum_t transA, dplasma_enum_t transB,
                           dplasma_complex64_t alpha, int Am, int An, int Aseed,
                                                    int Bm, int Bn, int Bseed,
                           dplasma_complex64_t beta,  int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *dcCfinal );

static int
parsec_core_gemm(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int transA;
    int transB;
    int m;
    int n;
    int k;
    dplasma_complex64_t alpha;
    dplasma_complex64_t *A;
    int lda;
    dplasma_complex64_t *B;
    int ldb;
    dplasma_complex64_t beta;
    dplasma_complex64_t *C;
    int ldc;

    parsec_dtd_unpack_args(this_task, &transA, &transB, &m, &n, &k, &alpha, &A,
                           &lda, &B, &ldb, &beta, &C, &ldc);

    CORE_zgemm(transA, transB, m, n, k,
               alpha,  A, lda,
                       B, ldb,
               beta,   C, ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872;
    int Bseed = 4674;
    int Cseed = 2873;
    int tA = dplasmaNoTrans;
    int tB = dplasmaNoTrans;
    dplasma_complex64_t alpha =  0.51;
    dplasma_complex64_t beta  = -0.42;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
    iparam[IPARAM_NGPUS] = DPLASMA_ERR_NOT_SUPPORTED;

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    PASTE_CODE_FLOPS(FLOPS_ZGEMM, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    int tile_longer_dim = MB;
    int tile_shorter_dim = NB;
    /* printf("MB: longer tile dim\tNB: shorter tile dim\n");*/
    /* printf("tile_longer_dim=%d\ttile_shorter_dim=%d\n",tile_longer_dim,tile_shorter_dim);*/

    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Tile,
                               rank, tile_shorter_dim, tile_shorter_dim, LDC, N, 0, 0,
                               M, N, P, nodes/P, KP, KQ, IP, JQ));

    /* Initializing dc for dtd */
    two_dim_block_cyclic_t *__dcC = &dcC;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcC);

    /* initializing matrix structure */
    if(!check)
    {
        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                   rank, tile_shorter_dim, tile_longer_dim, LDA, K, 0, 0,
                                   M, K, P, nodes/P, KP, KQ, IP, JQ));

        /* Initializing dc for dtd */
        two_dim_block_cyclic_t *__dcA = &dcA;
        parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

        PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
            two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                   rank, tile_longer_dim, tile_shorter_dim, LDB, N, 0, 0,
                                   K, N, P, nodes/P, KP, KQ, IP, JQ));

        /* Initializing dc for dtd */
        two_dim_block_cyclic_t *__dcB = &dcB;
        parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcB);

        /* Getting new parsec handle of dtd type */
        parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

        /* Default type */
        parsec_arena_datatype_t *tile_full = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
        dplasma_add2arena_tile( tile_full,
                                dcA.super.mb*dcA.super.nb*sizeof(dplasma_complex64_t),
                                PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_complex_t, dcA.super.mb );

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed);
        if(loud > 2) printf("Done\n");

        int m, n, k;
        int ldam, ldak, ldbn, ldbk, ldcm;
        int tempmm, tempnn, tempkn;
        int ic, jc; /* ic, jc are counters for the 7-level implementation */

        dplasma_complex64_t zbeta;
        dplasma_complex64_t zone = (dplasma_complex64_t)1.0;

        parsec_context_add_taskpool( parsec, dtd_tp );

        #define MDEV_N_SKIP 4
        #define MDEV_M_SKIP 4
        SYNC_TIME_START();

        /* #### parsec context Starting #### */

        /* start parsec context */
        parsec_context_start(parsec);
        if( ( tA == dplasmaNoTrans ) && ( tB == dplasmaNoTrans ) )
        {
            for( jc = 0; jc < dcC.super.nt; jc += MDEV_N_SKIP ) {

                for( k = 0; k < dcA.super.nt; k++ ) {
                    zbeta = k == 0 ? beta : zone;
                    ldbk = BLKLDD(&dcB.super, k);
                    tempkn = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;

                    for( ic = 0; ic < dcC.super.mt; ic+= MDEV_M_SKIP ) {

                        for( n = jc; n < jc+MDEV_N_SKIP && n < dcC.super.nt; n++ ) {
                        tempnn = n == dcC.super.nt-1 ? dcC.super.n-n*dcC.super.nb : dcC.super.nb;

                            for( m = ic; m < ic+MDEV_M_SKIP && m < dcC.super.mt; m++ ) {
                                tempmm = m == dcC.super.mt-1 ? dcC.super.m-m*dcC.super.mb : dcC.super.mb;
                                ldcm = BLKLDD(&dcC.super, m);
                                ldam = BLKLDD(&dcA.super, m);

                                /* printf("DEBUG:\tjc=%d\tk=%d\tic=%d\tn=%d\tm=%d\n", jc, k, ic, n, m); */
                                /* printf("DEBUG (cont):\ntempmm=%d\ntempnn=%d\ntempkn=%d\nldam=%d\nldbk=%d\nldcm=%d\n\n",
                                                tempmm,    tempnn,    tempkn,    ldam,    ldbk,    ldcm); */
                                parsec_dtd_insert_task( dtd_tp,  &parsec_core_gemm, 0, PARSEC_DEV_CPU,  "Gemm",
                                            sizeof(int),           &tA,                           PARSEC_VALUE,
                                            sizeof(int),           &tB,                           PARSEC_VALUE,
                                            sizeof(int),           &tempmm,                       PARSEC_VALUE,
                                            sizeof(int),           &tempnn,                       PARSEC_VALUE,
                                            sizeof(int),           &tempkn,                       PARSEC_VALUE,
                                            sizeof(dplasma_complex64_t),           &alpha,         PARSEC_VALUE,
                                            PASSED_BY_REF,     PARSEC_DTD_TILE_OF(A, m, k),     PARSEC_INPUT | TILE_FULL,
                                            sizeof(int),           &ldam,                         PARSEC_VALUE,
                                            PASSED_BY_REF,     (long)PARSEC_DTD_TILE_OF(B, k, n),     PARSEC_INPUT | TILE_FULL,
                                            sizeof(int),           &ldbk,                         PARSEC_VALUE,
                                            sizeof(dplasma_complex64_t),           &zbeta,         PARSEC_VALUE,
                                            PASSED_BY_REF,     (long)PARSEC_DTD_TILE_OF(C, m, n),     PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                            sizeof(int),           &ldcm,                         PARSEC_VALUE,
                                                    PARSEC_DTD_ARG_END );
                            
                            }
                        }
                    }
                }
                // parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );
                // parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcB );
                // parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcC );
                // parsec_dtd_taskpool_wait( dtd_tp );
            }
        }
        else
        {
            fprintf(stderr, "Not supported. Only run with no transpose.\n\n");
        }


        parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );
        parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcB );
        parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcC );

        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_taskpool_wait( dtd_tp );

        /* Waiting on all handle and turning everything off for this context */
        parsec_context_wait( parsec );

        /* #### PaRSEC context is done #### */

        SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d MB= %4d N= %7d : %14f gflops\n",
                               P, Q, NB, MB, N,
                               gflops=(flops/1e9)/sync_time_elapsed));

        /* Cleaning up the parsec handle */
        parsec_taskpool_free( dtd_tp );

        /* Cleaning data arrays we allocated for communication */
        parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
        dplasma_matrix_del2arena( tile_full );
        parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

        parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcB );
        parsec_data_free(dcB.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    } else {
        int Am, An, Bm, Bn;
        PASTE_CODE_ALLOCATE_MATRIX(dcC2, check,
            two_dim_block_cyclic, (&dcC2, matrix_ComplexDouble, matrix_Tile,
                                   rank, tile_shorter_dim, tile_shorter_dim, LDC, N, 0, 0,
                                   M, N, P, nodes/P, KP, KQ, IP, JQ));

        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC2, Cseed);

#if defined(PRECISION_z) || defined(PRECISION_c)
        for(tA=0; tA<1; tA++) { /* return to 0-2 when transpose added */
            for(tB=0; tB<1; tB++) {
#else
        for(tA=0; tA<1; tA++) { /* return to 0-1 when transpose added */
            for(tB=0; tB<1; tB++) {
#endif

                /* Getting new parsec handle of dtd type */
                parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

                if ( trans[tA] == dplasmaNoTrans ) {
                    Am = M; An = K;
                } else {
                    Am = K; An = M;
                }
                if ( trans[tB] == dplasmaNoTrans ) {
                    Bm = K; Bn = N;
                } else {
                    Bm = N; Bn = K;
                }

                PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
                    two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                           rank, tile_shorter_dim, tile_longer_dim, LDA, LDA, 0, 0,
                                           Am, An, P, nodes/P, KP, KQ, IP, JQ));

                /* Initializing dc for dtd */
                two_dim_block_cyclic_t *__dcA = &dcA;
                parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

                PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
                    two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                           rank, tile_longer_dim, tile_shorter_dim, LDB, LDB, 0, 0,
                                           Bm, Bn, P, nodes/P, KP, KQ, IP, JQ));

                /* Initializing dc for dtd */
                two_dim_block_cyclic_t *__dcB = &dcB;
                parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcB);

                /* Allocating data arrays to be used by comm engine */
                /* Default type */
                parsec_arena_datatype_t *tile_full = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
                dplasma_add2arena_tile( tile_full,
                                        dcA.super.mb*dcA.super.nb*sizeof(dplasma_complex64_t),
                                        PARSEC_ARENA_ALIGNMENT_SSE,
                                        parsec_datatype_double_complex_t, dcA.super.mb );

                dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
                dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed);

                if ( rank == 0 ) {
                    printf("***************************************************\n");
                    printf(" ----- TESTING ZGEMM (%s, %s) -------- \n",
                           transstr[tA], transstr[tB]);
                }

                /* matrix generation */
                if(loud) printf("Generate matrices ... ");
                dplasma_zlacpy( parsec, dplasmaUpperLower,
                                (parsec_tiled_matrix_dc_t *)&dcC2, (parsec_tiled_matrix_dc_t *)&dcC );
                if(loud) printf("Done\n");

                /* Create GEMM PaRSEC */
                if(loud) printf("Compute ... ... ");

                int m, n, k;
                int ldam, ldak, ldbn, ldbk, ldcm;
                int tempmm, tempnn, tempkn, tempkm;
                int ic, jc; /* ic, jc are counters for the 7-level implementation */

                dplasma_complex64_t zbeta;
                dplasma_complex64_t zone = (dplasma_complex64_t)1.0;

                /* Registering the handle with parsec context */
                parsec_context_add_taskpool( parsec, dtd_tp );

                SYNC_TIME_START();

                /* #### parsec context Starting #### */

                /*
                 * Daniel notes.
                 * Most notes recorded in worklog.txt
                 * This approach still naievely uses tiles at their most base level.
                 * Maybe there is a better way. This we will look into next.
                 */

                /* start parsec context */
                #define MDEV_N_SKIP 4
                #define MDEV_M_SKIP 4
                parsec_context_start(parsec);
                if( ( trans[tA] == dplasmaNoTrans ) && ( trans[tB] == dplasmaNoTrans ) )
                {
                    for( jc = 0; jc < dcC.super.nt; jc += MDEV_N_SKIP ) {

                        for( k = 0; k < dcA.super.nt; k++ ) {
                            zbeta = k == 0 ? beta : zone;
                            ldbk = BLKLDD(&dcB.super, k);
                            tempkn = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;

                            for( ic = 0; ic < dcC.super.mt; ic+= MDEV_M_SKIP ) {

                                for( n = jc; n < jc+MDEV_N_SKIP && n < dcC.super.nt; n++ ) {
                                tempnn = n == dcC.super.nt-1 ? dcC.super.n-n*dcC.super.nb : dcC.super.nb;

                                    for( m = ic; m < ic+MDEV_M_SKIP && m < dcC.super.mt; m++ ) {
                                        tempmm = m == dcC.super.mt-1 ? dcC.super.m-m*dcC.super.mb : dcC.super.mb;
                                        ldcm = BLKLDD(&dcC.super, m);
                                        ldam = BLKLDD(&dcA.super, m);

                                        /* printf("DEBUG:\tjc=%d\tk=%d\tic=%d\tn=%d\tm=%d\n", jc, k, ic, n, m); */
                                        /* printf("DEBUG (cont):\ntempmm=%d\ntempnn=%d\ntempkn=%d\nldam=%d\nldbk=%d\nldcm=%d\n\n",
                                                        tempmm,    tempnn,    tempkn,    ldam,    ldbk,    ldcm); */
                                        parsec_dtd_insert_task( dtd_tp,  &parsec_core_gemm, 0, PARSEC_DEV_CPU,  "Gemm",
                                                    sizeof(int),           &trans[tA],                    PARSEC_VALUE,
                                                    sizeof(int),           &trans[tB],                    PARSEC_VALUE,
                                                    sizeof(int),           &tempmm,                       PARSEC_VALUE,
                                                    sizeof(int),           &tempnn,                       PARSEC_VALUE,
                                                    sizeof(int),           &tempkn,                       PARSEC_VALUE,
                                                    sizeof(dplasma_complex64_t),           &alpha,         PARSEC_VALUE,
                                                    PASSED_BY_REF,     PARSEC_DTD_TILE_OF(A, m, k),     PARSEC_INPUT | TILE_FULL,
                                                    sizeof(int),           &ldam,                         PARSEC_VALUE,
                                                    PASSED_BY_REF,     (long)PARSEC_DTD_TILE_OF(B, k, n),     PARSEC_INPUT | TILE_FULL,
                                                    sizeof(int),           &ldbk,                         PARSEC_VALUE,
                                                    sizeof(dplasma_complex64_t),           &zbeta,         PARSEC_VALUE,
                                                    PASSED_BY_REF,     (long)PARSEC_DTD_TILE_OF(C, m, n),     PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                                    sizeof(int),           &ldcm,                         PARSEC_VALUE,
                                                            PARSEC_DTD_ARG_END );
                                    
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    fprintf(stderr, "Not supported. Only run with no transpose.\n\n");
                }

                parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );
                parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcB );
                parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcC );

                /* finishing all the tasks inserted, but not finishing the handle */
                parsec_dtd_taskpool_wait( dtd_tp );

                /* Waiting on all handle and turning everything off for this context */
                parsec_context_wait( parsec );

                if(loud) printf("Done\n");

                /* #### PaRSEC context is done #### */

                /* Cleaning up the parsec handle */
                parsec_taskpool_free( dtd_tp );

                /* Cleaning data arrays we allocated for communication */
                parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
                dplasma_matrix_del2arena( tile_full );
                parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );

                parsec_data_free(dcA.mat);
                parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

                parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcB );

                parsec_data_free(dcB.mat);
                parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);

                /* Check the solution */
                info_solution = check_solution( parsec, (rank == 0) ? loud : 0,
                                                trans[tA], trans[tB],
                                                alpha, Am, An, Aseed,
                                                       Bm, Bn, Bseed,
                                                beta,  M,  N,  Cseed,
                                                &dcC);
                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZGEMM (%s, %s) ...... PASSED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    else {
                        printf(" ---- TESTING ZGEMM (%s, %s) ... FAILED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    printf("***************************************************\n");
                }
            }
        }
        parsec_data_free(dcC2.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC2);
    }

    /* Cleaning data arrays we allocated for communication */
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcC );

    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);

    cleanup_parsec(parsec, iparam);

    return info_solution;
}

/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( parsec_context_t *parsec, int loud,
                           dplasma_enum_t transA, dplasma_enum_t transB,
                           dplasma_complex64_t alpha, int Am, int An, int Aseed,
                                                    int Bm, int Bn, int Bseed,
                           dplasma_complex64_t beta,  int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *dcCfinal )
{
    int info_solution = 1;
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int K  = ( transA == dplasmaNoTrans ) ? An : Am ;
    int MB = dcCfinal->super.mb;
    int NB = dcCfinal->super.nb;
    int LDA = Am;
    int LDB = Bm;
    int LDC = M;
    int rank  = dcCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Lapack,
                               rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1, 1, 0, 0));
    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Lapack,
                               rank, MB, NB, LDB, Bn, 0, 0,
                               Bm, Bn, 1, 1, 1, 1, 0, 0));
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Lapack,
                               rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1, 1, 0, 0));

    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed );
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed );
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed );

    Anorm        = dplasma_zlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcA );
    Bnorm        = dplasma_zlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcB );
    Cinitnorm    = dplasma_zlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcC );
    Cdplasmanorm = dplasma_zlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_dc_t*)dcCfinal );

    if ( rank == 0 ) {
        cblas_zgemm(CblasColMajor,
                    (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                    M, N, K,
                    CBLAS_SADDR(alpha), dcA.mat, LDA,
                                        dcB.mat, LDB,
                    CBLAS_SADDR(beta),  dcC.mat, LDC );
    }

    Clapacknorm = dplasma_zlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcC );

    dplasma_zgeadd( parsec, dplasmaNoTrans, -1.0, (parsec_tiled_matrix_dc_t*)dcCfinal,
                                           1.0, (parsec_tiled_matrix_dc_t*)&dcC );

    Rnorm = dplasma_zlange( parsec, dplasmaMaxNorm, (parsec_tiled_matrix_dc_t*)&dcC);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||B||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*B+b*C)||_inf = %e, ||dplasma(a*A*B+b*C)||_inf = %e, ||R||_m = %e\n",
                   Anorm, Bnorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm);
        }

        result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * max(M,N) * eps);
        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcB.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);

    return info_solution;
}
