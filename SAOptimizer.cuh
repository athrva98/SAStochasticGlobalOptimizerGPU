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
                unsigned int seed, NumericType* vectors, NumericType* objective_values)
        : obj_func(obj_func), temperature(temperature), num_params(num_params),
          base_seed(seed), vectors(vectors), objective_values(objective_values) {}

    __device__
    void operator()(int idx) {
        unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        thrust::default_random_engine rng(base_seed + thread_id);
        thrust::uniform_real_distribution<NumericType> dist(0.0f, 1.0f);

        // Calculate the objective value for the current vector
        NumericType current_value = obj_func(&vectors[idx * num_params]);

        // Probabilistic acceptance criterion
        if (shouldAccept(current_value, objective_values[idx], rng, dist)) {
            objective_values[idx] = current_value;
            for (int i = 0; i < num_params; ++i) {
                vectors[idx * num_params + i] = vectors[idx * num_params + i]; // Update parameters if accepted
            }
        }
    }

private:
    ObjectiveFunction obj_func;
    NumericType temperature;
    size_t num_params;
    unsigned int base_seed;
    NumericType* vectors;
    NumericType* objective_values;

    __device__
    bool shouldAccept(NumericType new_value, NumericType old_value,
                      thrust::default_random_engine& rng, thrust::uniform_real_distribution<NumericType>& dist) {
        if (new_value < old_value) {
            return true;
        } else {
            NumericType acceptance_probability = exp(-abs(old_value - new_value) / temperature);
            return dist(rng) < acceptance_probability;
        }
    }
};
