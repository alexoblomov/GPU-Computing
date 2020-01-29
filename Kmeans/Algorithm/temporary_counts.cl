void counts(__global int *A,  __global float *result, const uint n_points) {
    uint thread_id = get_local_id(0);
    uint p_id = get_global_id(0);
    
    result[A[p_id],thread_id] = 1
}