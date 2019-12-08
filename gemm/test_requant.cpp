#include <stdio.h>
#include <vector>
using namespace std;
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "gemm_symm_int8.h"

extern "C" {
    void int8kernel_m4(int32_t* dst, const int8_t* sa, const int8_t* sb, size_t k, size_t n, size_t ldc);
    void int8kernel_m2(int32_t* dst, const int8_t* sa, const int8_t* sb, size_t k, size_t n, size_t ldc);
    void int8kernel_m1(int32_t* dst, const int8_t* sa, const int8_t* sb, size_t k, size_t n, size_t ldc);
    void reorder_a(const int8_t *a, int8_t *sa, int m, int k, int ldx);
    void int8kernel_m4_requant(int8_t* dst, const int8_t* sa, const int8_t* sb, int k, int n, int ldc, float* scales, const float* bias);
    void int8kernel_m2_requant(int8_t* dst, const int8_t* sa, const int8_t* sb, int k, int n, int ldc, float* scales, const float* bias);
    void int8kernel_m1_requant(int8_t* dst, const int8_t* sa, const int8_t* sb, int k, int n, int ldc, float* scales, const float* bias);
}

static inline void* fastMalloc(int size){
        void* ptr = 0;
        int iRet = posix_memalign(&ptr, 64, size);
        assert(0 == iRet);
        return ptr;
}


int test(int m, int k, int n, int ldc) {
    fprintf(stdout, "\n---- begin test m: %d, k: %d, n: %d\n", m, k, n);
    int8_t* a     = (int8_t*)fastMalloc(2*m*k);    
    int8_t* sa    = (int8_t*)fastMalloc(2*m*k);    

    int8_t* b  = (int8_t*)fastMalloc(2*k*n);
    int8_t* sb = (int8_t*)fastMalloc(2*k*n);

    int8_t* c  = (int8_t*)fastMalloc(2*m*n);
    int8_t* sc = (int8_t*)fastMalloc(2*m*n);

    for (int i = 0; i < m*n; ++i) {
        c[i] = sc[i] = 0;
    }

    int8_t val = 4;
    for (int i = 0; i < m*k; i++) {
        a[i] = val++;
    }
    fprintf(stdout, "----- print a:\n");
    // print_int8_matrix(a, m, k, k);

    val = 3;
    for (int i = 0; i < k*n; i++) {
        b[i] = val++;
    }
    fprintf(stdout, "----- print b:\n");
    // print_int8_matrix(b, k, n, n);

    
    reorder_a(a, sa, m, k, k);

    reorder_b(b, sb, k, n, n);
    fprintf(stdout, "** reorder_b finish\n");

    std::vector<float> scales(m);
    std::vector<float> bias(m);

    for (int i = 0; i < m; ++i) {
        scales[i] = 1.0f;
        bias[i] = 1.0f;
    }

    float *sptr = scales.data();
    float *bptr = bias.data();

    {
        int8_t* pa = sa;
        int8_t* pb = sb;
        int8_t* pc = c;
        const int nn = (m >> 2) << 2;
        for (int i = 0; i < nn; i += 4) {
            int8kernel_m4_requant(pc + i * ldc, pa + i * k, pb, k, n, n, sptr + i, bptr + i);
        }
        pa += nn * k;
        pb += nn * n;
        sptr += nn;
        bptr += nn;

        switch(m-nn)
        {
            case 3:
                int8kernel_m2_requant(pc, pa, pb, k, n, n, sptr, bptr);
                pc += 2 * n;
                pa += 2 * k;
                sptr += 2;
                bptr += 2;
                int8kernel_m1_requant(pc, pa, pb, k, n, n, sptr, bptr);
                break;
            case 2:
                int8kernel_m2_requant(pc, pa, pb, k, n, n, sptr, bptr);
                break;
            case 1:
                int8kernel_m1_requant(pc, pa, pb, k, n, n, sptr, bptr);
                break;
            case 0:
            default:
                break;
        }
    }
    fprintf(stdout, "----- print baseline:\n");
    // print_int32_matrix(c, m, n, n);
//    fprintf(stdout, "3. compute_baseline finish\n");

    fprintf(stdout, "\n---- test m: %d, k: %d, n: %d\n", m, k, n);

    sptr = scales.data();
    bptr = bias.data();
    int8kernel((void*)sc, sa, sb, m, k, n, n, sptr, bptr);
    fprintf(stdout, "** compute_new finish\n");
    fprintf(stdout, "----- print myresult:\n");
    // print_int32_matrix(sc, m, n, n);

    for (int idx = 0; idx < m*n; ++idx) {
        if (c[idx] != sc[idx]) {
                fprintf(stdout, " int8kernel err in idx: %d, c[idx]: %d, sc[idx]: %d", idx, c[idx], sc[idx]);    
                exit(-1);
        
        }
    }
//    for (int i = 0; i < m; ++i) {
//        for (int j = 0; j < n; ++j) {
//            int idx = i * n + j;
//            if (c[idx] != sc[idx]) {
//                fprintf(stdout, " int8kernel err in [%d, %d], idx: %d, c[idx]: %d, sc[idx]: %d", i, j, idx, c[idx], sc[idx]);    
//                exit(-1);
//            }
//        }
//    }
    fprintf(stdout, "** check result passed\n");

    free(a);
    free(sa);

    free(sb);
    free(b);

    free(c);
    free(sc);
    return 0;
}

int main() {
   // test m k n
   // for (int kk = 1; kk < 25; ++kk) {
   //     test(4, kk, 1, 1);
   // }

   //  for (int nn = 1; nn < 15; ++nn) {
   //      test(4, 1, nn, nn);
   //  }
    test(1, 1, 17, 17);
    for (int m = 1; m < 50; m++)
 for(int n = 1; n < 50; ++n) {
     for (int k = 1; k < 50; ++k) {
         test(8, k, n, n);        
    }
 }
    return 0;
}
