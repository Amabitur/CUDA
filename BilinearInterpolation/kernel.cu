texture<unsigned int, 2, cudaReadModeElementType> d_texture_interp_float;

__global__ void interpolate(unsigned int * __restrict__ d_result, const int M1, const int M2, const int N1, const int N2)
{
    const int l = threadIdx.x + blockDim.x * blockIdx.x;
    const int k = threadIdx.y + blockDim.y * blockIdx.y;

    float x = (float(l)/N1)*M1;
    float y = (float(k)/N2)*M2;
    if ((l<N1)&&(k<N2)) { d_result[l*N1 + k] = tex2D(d_texture_interp_float, y, x); }
}
