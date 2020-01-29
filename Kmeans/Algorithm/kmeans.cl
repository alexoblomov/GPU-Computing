__kernel void _kmeans(const uint n_points, const uint n_clusters, const uint max_iter, const uint dim,    
    __global float* P, __global float* C, __global int* A)
{
    // creating temporary arrays for counts and sums
    int counts[n_clusters] = {0};
    int tmp_counts[n_clusters][n_points] = {{0}, {0}};
    float tmp_sums[n_clusters][n_points][dim] = {{0}, {0}};


}