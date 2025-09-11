#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

__kernel void dpp_impl(__global const float *inp_mat, const int M, __global int *output_id,
                              int iteration, __global float *cis, __global float *di2s,
                              const float numerical_threshold, const int selected_token_num,
                              __global int *output_ids,
                              __local float *local_max_array,
                              __local int *local_max_ids)
{
    uint batch_idx = get_global_id(0);
    uint gid_1 = get_global_id(1);

    // Step 1: argmax
    uint lid_0 = get_local_id(1);
    uint local_size = get_local_size(1);

    // Initialize local maximum value to a very small value
    float my_local_max = -FLT_MAX;
    uint my_local_id = 0;
    __global const float* pdata = di2s + batch_idx * M;

    for (int i = lid_0; i < M; i += local_size) {
        if (pdata[i] > my_local_max) {
            my_local_max = pdata[i];
            my_local_id = i;
        }
    }

    // Write the local maximum found by each work item to shared local memory
    local_max_array[lid_0] = my_local_max;
    local_max_ids[lid_0] = my_local_id;

    // Synchronize to ensure all work items have completed the first phase write
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 2: Parallel reduction to find the final maximum in local memory ---
    // This loop will continuously halve until local_max_array[0] contains the final result
    for (size_t s = local_size / 2; s > 0; s >>= 1) {
        // Compare the value at current position with the value at another position
        if (lid_0 < s && local_max_array[lid_0 + s] > local_max_array[lid_0] ) {
            local_max_array[lid_0] = local_max_array[lid_0 + s];
            local_max_ids[lid_0] = local_max_ids[lid_0 + s];
        }
        // Synchronize to ensure data is visible to all threads after this round of comparison
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Step 2: update orthogonal vector.
    if (gid_1 >= M)
        return;

    const int selected_idx = local_max_ids[0];
    size_t offset = batch_idx * M * M;
    size_t total_tokens = M;
    __global const float *kernel_data = inp_mat + offset;
    __global float *di2s_data = di2s + batch_idx * M;
    __global float *cis_data = cis + batch_idx * M * selected_token_num;
    __global int* output_ids_data = output_ids + batch_idx * selected_token_num;

    // Get the normalization factor
    float norm_factor = sqrt(di2s_data[selected_idx] + numerical_threshold);

    // Compute the new orthogonal vector for each token
    size_t j = gid_1;

    size_t kernel_idx = selected_idx * total_tokens + j;
    float kernel_val = kernel_data[kernel_idx];

    // Subtract the projection onto previously selected vectors
    // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
    float projection = 0.0f;
    __global float *cis_selected_t = cis_data + selected_token_num * selected_idx;
    __global float *cis_t = cis_data + selected_token_num * j;

    __attribute__((opencl_unroll_hint(4))) for (size_t prev_t = 0; prev_t < iteration; ++prev_t)
    {
        projection += cis_selected_t[prev_t] * cis_t[prev_t];
    }

    // Store the orthogonalized vector element
    size_t cis_current_idx = iteration + j * selected_token_num;
    cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;

    // step 3: update_marginal_gains
    size_t cis_idx = iteration + j * selected_token_num;
    float eis_j = cis_data[cis_idx];

    // Subtract the squared orthogonal component
    if (selected_idx == j) {
        di2s_data[selected_idx] = -INFINITY;
        output_ids_data[iteration] = selected_idx;
    }
    else {
        di2s_data[j] -= eis_j * eis_j;
    }
}