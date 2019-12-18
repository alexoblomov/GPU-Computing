__kernel void reduce(__global float *a,
                     __global float *r,
                     __local float *b)
{
    uint gid = get_global_id(0);
    uint wid = get_group_id(0);
    uint lid = get_local_id(0);
    uint gs = get_local_size(0);

    b[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = gs/2; s > 0; s >>= 1) {
        if(lid < s) {
          b[lid] += b[lid+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0) r[wid] = b[lid];
}
