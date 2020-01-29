#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 1
#endif

// method 2 of Reduction/reduction_sum.cl

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce(__global float *input,  __global float *result, __local volatile float *localBuffer, const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);

    localBuffer[lid] = input[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(uint s=1; s < WG_SIZE; s *= 2) {
        int index = 2 * s * lid;
        if (index < WG_SIZE) {
            localBuffer[index] += localBuffer[index + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write result for this block to global mem
    if (lid == 0) result[get_group_id(0)] = localBuffer[0];
}