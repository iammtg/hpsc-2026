#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  // Load
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);

  // 比較用index
  __m512i jvec = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
  for(int i=0; i<N; i++) {
    // mask (i!=j)
    __m512i ivec = _mm512_set1_epi32(i);
    __mmask16 mask = _mm512_cmpneq_epi32_mask(ivec, jvec);

     // rx = x[i] - x[j], ry = y[i] - y[j]
    __m512 rx = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);
    __m512 ry = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);

    // 1/r = 1/sqrt(rx^2 + ry^2)
    __m512 rinv  = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx),_mm512_mul_ps(ry, ry)));

    // 1/r^3
    __m512 rinv3 = _mm512_mul_ps(_mm512_mul_ps(rinv, rinv), rinv);

    // m[j] / r^3
    __m512 scale = _mm512_mul_ps(mvec, rinv3);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, _mm512_mul_ps(rx, scale));
    fy[i] -= _mm512_mask_reduce_add_ps(mask, _mm512_mul_ps(ry, scale));

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
