#include <cuda_runtime.h>
#include <limits>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

typedef float NumericType;

template<typename ObjectiveFunction, typename NumericType>
class SAOptimizer {
public:
    SAOptimizer(ObjectiveFunction obj_func, NumericType temperature,
        NumericType& shared_optimal_value,
        NumericType* shared_optimal_parameters,
        size_t num_params,
        unsigned int seed)
        : obj_func(obj_func), temperature(temperature),
        shared_optimal_value(shared_optimal_value),
        shared_optimal_parameters(shared_optimal_parameters),
        num_params(num_params), base_seed(seed) {}

    __host__ __device__
    void operator()(int idx, NumericType* vectors, NumericType* objective_values) {

        unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        thrust::default_random_engine eng(base_seed + thread_id);
        thrust::uniform_real_distribution<NumericeType> dist(0.0f, 1.0f);

        // Calculate the objective value for the current vector
        NumericType current_value = obj_func(&vectors[idx * num_params]);

        // Probabilistic acceptance criterion
        if (shouldAccept(current_value, objective_values[idx], rng, dist)) {
            objective_values[idx] = current_value;
            atomic_update_best_parameters(&vectors[idx * num_params], num_params, current_value, shared_optimal_parameters);
        }
    }

private:
    ObjectiveFunction obj_func;
    NumericType temperature;
    NumericType& shared_optimal_value;
    NumericType* shared_optimal_parameters;
    size_t num_params;
    unsigned int base_seed;

    __host__ __device__
    bool shouldAccept(NumericType new_value, NumericType old_value,
                thrust::default_random_engine& rng, thrust::uniform_real_distribution<NumericeType>& dist) {
        if (new_value < old_value) {
            return true;
        } else {
            NumericType acceptance_probability = exp(-1 * abs(old_value - new_value) / temperature);
            return dist(rng) < acceptance_probability;
        }
    }

    __device__
    void atomic_update_best_parameters(NumericType* params, int num_params, NumericType new_value, NumericType* shared_optimal_parameters) {
        // First, we need to atomically update the best objective value
        NumericType* best_value_address = &shared_optimal_parameters[0];
        float old_best_value = *best_value_address;
        float assumed;

        do {
            assumed = old_best_value;
            // If the new value is not better than the current best, then we don't update
            if (assumed <= new_value) {
                return;
            }
            // Try to update the best objective value atomically
            old_best_value = atomicCAS((unsigned int*)best_value_address, __float_as_uint(assumed), __float_as_uint(new_value));
        } while (__float_as_uint(assumed) != __float_as_uint(old_best_value));

        // Ensure that the atomic update is completed before proceeding
        __threadfence();

        // Check if the current thread has successfully updated the best objective value
        if (__float_as_uint(assumed) > __float_as_uint(new_value)) {
            // Now, update the best parameters
            // Use a simple spin-lock for the parameter update
            while (atomicCAS(&shared_optimal_parameters[1], 0, 1) != 0) {
                // Spin-wait until we acquire the lock
            }

            // We have the lock, update the parameters
            for (int i = 0; i < num_params; ++i) {
                shared_optimal_parameters[i+2] = params[i]; // Assuming first two elements are for value and lock
            }

            // Release the lock
            atomicExch(&shared_optimal_parameters[1], 0);

            // Ensure that the parameter update is visible to all threads
            __threadfence();
        }
    }
};

struct SimpleQuadraticObjectiveFunction {
    __host__ __device__
    NumericType operator()(NumericType* params) {
        return params[0] * params[0] + params[1] * params[1];
    }
}
