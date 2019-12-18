#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 1
#endif

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce1(__global float *input,  __global float *result, __local volatile float *localBuffer, const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);
    localBuffer[lid] = input[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(uint s=1; s < WG_SIZE; s *= 2) {
        if (lid % (2*s) == 0) { // Divergent and % is a slow operator
            localBuffer[lid] += localBuffer[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write result for this block to global mem
    if (lid == 0) result[get_group_id(0)] = localBuffer[0];
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce2(__global float *input,  __global float *result, __local volatile float *localBuffer, const uint n) {
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

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce3(__global float *input,  __global float *result, __local volatile float *localBuffer, const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);

    localBuffer[lid] = input[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for (uint s=WG_SIZE/2; s>0; s>>=1) {
        if (lid < s) {
            localBuffer[lid] += localBuffer[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write result for this block to global mem
    if (lid == 0) result[get_group_id(0)] = localBuffer[0];
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce4(__global float *input,  __global float *result, __local volatile float *localBuffer, const uint n) {
    // each thread loads two elements from global to shared mem
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    uint k = gid * (WG_SIZE*2) + lid;
    localBuffer[lid] = input[k] + input[k+WG_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
#pragma unroll 8
    for (uint s=WG_SIZE/2; s>0; s>>=1) {
        if (lid < s) {
            localBuffer[lid] += localBuffer[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write result for this block to global mem
    if (lid == 0) result[get_group_id(0)] = localBuffer[0];
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce5(__global float *input,  __global float *result, __local volatile float *localBuffer,const uint n) {
    const uint id = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group_size = get_local_size(0);
    const uint gid = get_group_id(0);

    localBuffer[lid] = input[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // local memory reduction
    uint i = group_size/2;
    for(; i>WAVEFRONT_SIZE; i >>= 1) {
        if(lid < i)
            localBuffer[lid] +=  localBuffer[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // wavefront reduction
    for(; i>0; i >>= 1) {
        if(lid < i)
            localBuffer[lid] +=  localBuffer[lid + i];
    }
    if(lid==0)
        result[gid] = localBuffer[0];
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce6(__global float *input,  __global float *result, __local volatile float *localBuffer, const uint n) {
    const uint id = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group_size = get_local_size(0);
    const uint gid = get_group_id(0);

    // Read input data
    localBuffer[lid] = input[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint offset = group_size / 2; offset > 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < offset) {
            localBuffer[lid] += localBuffer[lid + offset];
        }
    }

    // Write back group reduction to global memory
    if(lid==0)
        result[gid] = localBuffer[0];
}

void warpReduce(volatile __local float *sdata, uint tid);
inline void warpReduce(volatile __local float *sdata, uint tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce7(__global float *input,  __global float *result, __local volatile float *localBuffer, private uint n) {
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);

    const int grid_size = 2*get_global_size(0);
    uint k = gid * (WG_SIZE*2) + lid;

    /// initialize shared memory contents
    localBuffer[lid] = 0.;

    /// Reduce read
    while (k < n) {
        localBuffer[lid] += input[k] + input[k+WG_SIZE];
        k += grid_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (WG_SIZE >= 256)
    {
        if (lid < 128)
        {
            localBuffer[lid] += localBuffer[lid + 128];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (WG_SIZE >= 128)
    {
        if (lid < 64)
        {
            localBuffer[lid] += localBuffer[lid + 64];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /// Loop unrolling for the warp
    if (lid < 64) warpReduce(localBuffer, lid);

    /// Reduction in global memory
    if(lid==0)
        result[gid] = localBuffer[0];
}


__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduceMdb(__global float *input,  __global float *result, __local volatile float *localBuffer,  private uint n) {
    const uint id = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);

    localBuffer[lid] = input[id];

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < stride)
        {
            localBuffer[lid] += localBuffer[lid + stride];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0)
    {
        result[gid] = localBuffer[0];
    }
}