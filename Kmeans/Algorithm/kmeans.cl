#define TILE_WIDTH 16
__kernel void _kmeans(uint n_points, uint dim, uint n_clusters,
int bb_min, int bb_max, uint max_iter,
    __global float* d_P,
    __global float* d_C,
    __global int* d_A)
{
    int dx = get_local_id(0);
    int p_idx = dx * TILE_WIDTH;
    d_A[dx] = p_idx;
}