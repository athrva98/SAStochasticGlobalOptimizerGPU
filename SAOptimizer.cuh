#ifndef SAOPTIMIZER_CUH_
#define SAOPTIMIZER_CUH_
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>
#include <limits>
#include <iostream>


template<typename ObjectiveFunction, typename NumericType>
class SAOptimizer {
public:
    SAOptimizer(ObjectiveFunction obj_func, NumericType temperature, size_t num_params,
        unsigned int seed, NumericType* vectors, NumericType* objective_values,
        int total_evaluations, bool warm_start, float step_size)
        : obj_func(obj_func), temperature(temperature), num_params(num_params),
        base_seed(seed), vectors(vectors), objective_values(objective_values),
        total_evaluations(total_evaluations), warm_start(warm_start), step_size(step_size) {

        _allocate_old_params();
    }

    ~SAOptimizer() {
        if (old_params) {
            cudaFree(old_params);
        }
    }

    void optimize(int block_size) {
        int num_blocks = (total_evaluations + block_size - 1) / block_size;
        int num_threads = num_blocks * block_size;

        if (total_evaluations <= num_threads) {
            singleEvaluationKernel << <num_blocks, block_size >> > (obj_func, old_params, temperature,
                num_params, base_seed, vectors,
                objective_values, total_evaluations,
                step_size);
        }
        else {
            gridStrideKernel << <num_blocks, block_size >> > (obj_func, old_params, temperature,
                num_params, base_seed, vectors,
                objective_values, total_evaluations,
                step_size);
        }
    }

    __device__
        static bool shouldAccept(NumericType new_value, NumericType old_value, NumericType temperature,
            thrust::default_random_engine& rng, thrust::uniform_real_distribution<NumericType>& dist) {

        const NumericType epsilon = 1e-16;
        if (new_value < old_value && abs(new_value - old_value) > epsilon) {
            return true;
        }
        else {
            NumericType acceptance_probability = exp(-abs(old_value - new_value) / temperature);
            return dist(rng) < acceptance_probability;
        }
    }

private:
    ObjectiveFunction obj_func;
    NumericType temperature;
    size_t num_params;
    unsigned int base_seed;
    NumericType* vectors;
    NumericType* old_params; // to store the older params for reversion
    NumericType* objective_values;
    int total_evaluations;
    float step_size;
    bool warm_start;

    void _allocate_old_params() {
        cudaMalloc(&old_params, num_params * total_evaluations * sizeof(NumericType));
    }
};


template<typename ObjectiveFunction, typename NumericType>
__launch_bounds__(256 /* maxThreadsPerBlock */, 4 /* minBlocksPerMultiprocessor */)
__global__ void singleEvaluationKernel(ObjectiveFunction obj_func,
    NumericType* old_params, NumericType temperature,
    size_t num_params, unsigned int base_seed,
    NumericType* vectors, NumericType* objective_values,
    int total_evaluations, float step_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_evaluations) return;

    thrust::default_random_engine rng(base_seed + idx);
    thrust::uniform_real_distribution<NumericType> dist_step_size(-step_size, step_size);
    thrust::uniform_real_distribution<NumericType> dist(0.0f, 1.0f);

    // Copy of old parameters
    for (int i = 0; i < num_params; ++i) {
        old_params[idx * num_params + i] = vectors[idx * num_params + i];
    }

    // Apply perturbations to each parameter
    for (int i = 0; i < num_params; ++i) {
        vectors[idx * num_params + i] += dist_step_size(rng); // Apply perturbation
    }

    // Evaluate and decide whether to accept this new parameter set
    NumericType current_value = obj_func(&vectors[idx * num_params]);
    if (!SAOptimizer<ObjectiveFunction, NumericType>::shouldAccept(current_value, objective_values[idx], temperature, rng, dist)) {
        // Revert the changes if not accepted
        for (int i = 0; i < num_params; ++i) {
            vectors[idx * num_params + i] = old_params[idx * num_params + i];
        }
    }
    else {
        objective_values[idx] = current_value;
    }
}


template<typename ObjectiveFunction, typename NumericType>
__global__ void gridStrideKernel(ObjectiveFunction obj_func,
    NumericType* old_params, NumericType temperature,
    size_t num_params, unsigned int base_seed, NumericType* vectors,
    NumericType* objective_values, int total_evaluations, float step_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    thrust::default_random_engine rng(base_seed + thread_id);
    thrust::uniform_real_distribution<NumericType> dist_step_size(-step_size, step_size);
    thrust::uniform_real_distribution<NumericType> dist(0.0f, 1.0f);

    for (int idx = thread_id; idx < total_evaluations; idx += grid_size) {
        // Copy of old parameters
        for (int i = 0; i < num_params; ++i) {
            old_params[idx * num_params + i] = vectors[idx * num_params + i];
        }

        // Apply perturbations to each parameter
        for (int i = 0; i < num_params; ++i) {
            vectors[idx * num_params + i] += dist_step_size(rng); // Apply perturbation
        }

        // Evaluate and decide whether to accept this new parameter set
        NumericType current_value = obj_func(&vectors[idx * num_params]);
        if (!SAOptimizer<ObjectiveFunction, NumericType>::shouldAccept(current_value, objective_values[idx], temperature, rng, dist)) {
            // Revert the changes if not accepted
            for (int i = 0; i < num_params; ++i) {
                vectors[idx * num_params + i] = old_params[idx * num_params + i];
            }
        }
        else {
            objective_values[idx] = current_value;
        }
    }
}

#endif
