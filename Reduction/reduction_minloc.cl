#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 1
#endif

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce1(__global float *input,
             __global float *output_val, __global uint *output_loc,
             __local volatile float *local_val, __local volatile uint *local_loc,
             const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);
    uint gid = get_group_id(0);

    local_val[lid] = input[id];
    local_loc[lid] = id;

    float minval, testval;
    minval = local_val[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(uint s=1; s < WG_SIZE; s *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid % (2*s) == 0) {
            testval = local_val[lid + s];
            if (minval > testval)
            {
                local_val[lid] = testval;
                local_loc[lid] = local_loc[lid + s];
                minval = testval;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write result for this block to global mem
    if (lid == 0) {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    }
}

// Remove modulo to avoid branching
__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce2(__global float *input,
             __global float *output_val,
             __global uint *output_loc,
             __local volatile float *local_val,
             __local volatile uint *local_loc,
             const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);
    uint gid = get_group_id(0);

    local_val[lid] = input[id];
    local_loc[lid] = id;

    float minval, testval;


    for(uint s=1; s < WG_SIZE; s *= 2) {
        int index = 2 * s * lid;
        if (index < WG_SIZE) {
            minval = local_val[index];
            testval = local_val[index + s];
            if (minval > testval)
            {
                local_val[index] = testval;
                local_loc[index] = local_loc[index + s];
                minval = testval;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write result for this block to global mem
    if (lid == 0) {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    }
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce3(__global float *input,
             __global float *output_val, __global uint *output_loc,
             __local volatile float *local_val, __local volatile uint *local_loc,
             const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);
    uint gid = get_group_id(0);

    local_val[lid] = input[id];
    local_loc[lid] = id;
    float minval, testval;
    minval = local_val[lid];

    // do reduction in shared mem
    for (uint s=WG_SIZE/2; s>0; s>>=1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) {
            testval = local_val[lid + s];
            if (minval > testval)
            {
                local_val[lid] = testval;
                local_loc[lid] = local_loc[lid + s];
                minval = testval;
            }
        }
    }
    // write result for this block to global mem
    if (lid == 0) {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    }
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce4(__global float *input,
             __global float *output_val, __global uint *output_loc,
             __local volatile float *local_val, __local volatile uint *local_loc,
             const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    uint k = gid * (WG_SIZE*2) + lid;
    local_loc[lid] = input[k]< input[k+WG_SIZE] ? k : k+WG_SIZE;
    local_val[lid] = input[local_loc[lid]];
    float minval, testval;
    minval = local_val[lid];

    // do reduction in shared mem
    for (uint s=WG_SIZE/2; s>0; s>>=1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) {
            testval = local_val[lid + s];
            if (minval > testval)
            {
                local_val[lid] = testval;
                local_loc[lid] = local_loc[lid + s];
                minval = testval;
            }
        }
    }
    // write result for this block to global mem
    if (lid == 0) {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    }
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce5(__global float *input,
             __global float *output_val,
             __global uint *output_loc,
             __local volatile float *local_val,
             __local volatile uint *local_loc,
             const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);
    uint gid = get_group_id(0);

    local_val[lid] = input[id];
    local_loc[lid] = id;
    const uint group_size = get_local_size(0);
    float minval, testval;
    minval = local_val[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // local memory reduction
    uint i = group_size/2;
    for(; i>WAVEFRONT_SIZE; i >>= 1) {
        if(lid < i) {
            testval = local_val[lid + i];
            if (minval > testval)
            {
                local_val[lid] = testval;
                local_loc[lid] = local_loc[lid + i];
                minval = testval;
            }
            // localBuffer[lid] +=  localBuffer[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // wavefront reduction
    for(; i>0; i >>= 1) {
        if(lid < i) {
            testval = local_val[lid + i];
            if (minval > testval)
            {
                local_val[lid] = testval;
                local_loc[lid] = local_loc[lid + i];
                minval = testval;
            }
        }
    }
    // write result for this block to global mem
    if (lid == 0) {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    };
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce6(__global float *input,
             __global float *output_val,
             __global uint *output_loc,
             __local volatile float *local_val,
             __local volatile uint *local_loc,
             const uint n) {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint id = get_global_id(0);
    uint gid = get_group_id(0);
    const uint group_size = get_local_size(0);


    local_val[lid] = input[id];
    local_loc[lid] = id;

    float minval, testval;
    minval = local_val[lid];

    for(uint offset = group_size / 2; offset > 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < offset) {
            testval = local_val[lid + offset];
            if (minval > testval)
            {
                local_val[lid] = testval;
                local_loc[lid] = local_loc[lid + offset];
                minval = testval;
            }
        }
    }
    // write result for this block to global mem
    if (lid == 0) {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    };
}

void warpReduce(volatile __local float *sdata, volatile __local uint *pdata, uint tid);

inline void warpReduce(volatile __local float *sdata, volatile __local uint *pdata,uint lid) {
    pdata[lid] = sdata[lid] < sdata[lid+32] ? lid : lid+32;
    sdata[lid] = sdata[pdata[lid]];
    pdata[lid] = sdata[lid]< sdata[lid+16] ? lid : lid+16;
    sdata[lid] = sdata[pdata[lid]];
    pdata[lid] = sdata[lid]< sdata[lid+8] ? lid : lid+8;
    sdata[lid] = sdata[pdata[lid]];
    pdata[lid] = sdata[lid]< sdata[lid+4] ? lid : lid+4;
    sdata[lid] = sdata[pdata[lid]];
    pdata[lid] = sdata[lid]< sdata[lid+2] ? lid : lid+2;
    sdata[lid] = sdata[pdata[lid]];
    pdata[lid] = sdata[lid]< sdata[lid+1] ? lid : lid+1;
    sdata[lid] = sdata[pdata[lid]];
}

__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduce7(__global float *input,
             __global float *output_val,
             __global uint *output_loc,
             __local volatile float *local_val,
             __local volatile uint *local_loc,
             const uint n)
             {
    // each thread loads one element from global to shared mem
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    const int grid_size = 2*get_global_size(0);
    uint k = gid * (WG_SIZE*2) + lid;

    local_val[lid] = INFINITY;
    local_loc[lid] = k;
    float minval = INFINITY, tempval;
    uint minloc, temploc;
    /// Reduce read
    while (k < n)
    {
        temploc = input[k] < input[k+WG_SIZE] ? k : k+WG_SIZE;
        tempval = input[temploc];
        if (minval > tempval)
        {
            minval = tempval;
            minloc = temploc;
        }

        k += grid_size;
    }
    local_val[lid] = minval;
    local_loc[lid] = minloc;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (WG_SIZE >= 256)
    {
        if (lid < 128)
        {
            local_loc[lid] = local_val[lid] < local_val[lid+128] ? lid : lid+128;
            local_val[lid] = local_val[local_loc[lid]];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (WG_SIZE >= 128)
    {
        if (lid < 64)
        {
            local_loc[lid] = local_val[lid] < local_val[lid+64] ? lid : lid+64;
            local_val[lid] = local_val[local_loc[lid]];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /// Loop unrolling for the warp
    if (lid < 32) warpReduce(local_val, local_loc, lid);

    /// Reduction in global memory
    if(lid==0)
        // write result for this block to global mem
        if (lid == 0) {
            output_val[gid] = local_val[0];
            output_loc[gid] = local_loc[0];
        }
}


__kernel  __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduceMdb(__global float *input,
               __global float *output_val,
               __global uint *output_loc,
               __local volatile float *local_val,
               __local volatile uint *local_loc,
               const uint n) {

    const uint id = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);

    local_val[lid] = input[id];
    local_loc[lid] = id;

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < stride)
        {
            if (local_val[lid + stride] < local_val[lid]) {
                local_val[lid] = local_val[lid + stride];
                local_loc[lid] = local_loc[lid + stride];
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (lid == 0)
    {
        output_val[gid] = local_val[0];
        output_loc[gid] = local_loc[0];
    }
}

__kernel void findMinValue(__global float * myArray, __global double * mins, __global int * arraysize,__global int * elToWorkOn,__global int * dummy) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int groupID = get_group_id(0);
    int lsize = get_local_size(0);
    int gsize = get_global_id(0);
    int minloc = 0;
    int arrSize = *arraysize;
    int elPerGroup = *elToWorkOn;
    float mymin = INFINITY;


    __local float lmins[128];
//initialize local memory
    *(lmins + lid) = INFINITY;
    __local int lminlocs[128];

//this private value will reduce global memory access in the for loop (temp = *(myArray + i);)
    float temp;

//ofset and target of the for loop
    int offset = elPerGroup*groupID + lid;
    int target = elPerGroup*(groupID + 1);

//prevent that target<arrsize (may happen due to rounding errors or arrSize not a multiple of elPerGroup
    target = min(arrSize, target);

//find minimum for the kernel
//offset is different for each lid, leading to sequential memory access
    if (offset < arrSize) {
        for (int i = offset; i < target; i += lsize) {
            temp = *(myArray + i);
            if (temp < mymin) {
                mymin = temp;
                minloc = i;
            }
        }

        //store kernel minimum in local memory
        *(lminlocs + lid) = minloc;
        *(lmins + lid) = mymin;

        //find work group minimum (reduce global memory accesses)
        lsize = lsize >> 1;
        while (lsize > 0) {
            if (lid < lsize) {
                if (*(lmins + lid)> *(lmins + lid + lsize)) {
                    *(lmins + lid) = *(lmins + lid + lsize);
                    *(lminlocs + lid) = *(lminlocs + lid + lsize);
                }
            }
            lsize = lsize >> 1;
        }
    }
//write group minimum to global buffer
    if (lid == 0) {
        *(mins + groupID * 2 + 0) = *(lminlocs + 0);
        *(mins + groupID * 2 + 1) = *(lmins + 0);
    }
}

