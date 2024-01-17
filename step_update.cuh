#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

// Assuming _cons_prec_ntype is a macro for a constant precision numeric type
// It should be defined before using it, for example:
// #define _cons_prec_ntype float

template<typename ObjectiveFunction, typename NumericType>
class SAOptimizer {
public:
    SAOptimizer(ObjectiveFunction obj_func, NumericType temperature, NumericType& shared_optimal_value, NumericType* shared_optimal_parameters)
        : obj_func(obj_func), temperature(temperature), shared_optimal_value(shared_optimal_value), shared_optimal_parameters(shared_optimal_parameters), rng(), dist(0.0f, 1.0f) {}

    __host__ __device__
    void operator()(int idx, NumericType* vectors, NumericType* objective_values) {
        // calculate the objective value for the current vector
        NumericType current_value = obj_func(vectors[idx]);

        // Probabilistic acceptance criterion
        if (shouldAccept(current_value, objective_values[idx])) {
            objective_values[idx] = current_value;
            atomic_update_best_parameters(vectors[idx], current_value);
        }
    }

private:
    ObjectiveFunction obj_func;
    NumericType temperature;
    NumericType& shared_optimal_value;
    NumericType* shared_optimal_parameters;
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<NumericType> dist;

    __host__ __device__
    bool shouldAccept(NumericType new_value, NumericType old_value) {
        if (new_value < old_value) {
            return true;
        } else {
            NumericType acceptance_probability = exp(-1 * abs(old_value - new_value) / temperature);
            return dist(rng) < acceptance_probability;
        }
    }

    __device__ void atomic_update_best_parameters(float *address, float *params, int num_params, float new_value, float *best_params) {

        // First, we need to atomically update the best objective value
        int *address_as_int = (int*)address;
        int old = *address_as_int;
        int assumed;

        do {
            assumed = old;
            // If the new value is not better than the current best, then we don't update
            if (__int_as_float(assumed) <= new_value) {
                return;
            }
            // Try to update the best objective value atomically
            old = atomicCAS(address_as_int, assumed, __float_as_int(new_value));
        } while (assumed != old);

        // Ensure that the atomic update is completed before proceeding
        __threadfence();

        // Check if the current thread has successfully updated the best objective value
        if (__int_as_float(assumed) > new_value) {
            // Now, update the best parameters
            // This part is critical and needs to be done one thread at a time
            // Use atomicExch to set a lock (could be a separate lock array or a designated value)
            // Here, we use the first parameter as a simple lock by setting it to a special value
            float lock_val = -1.0f; // Assuming this value is never used in normal operation
            while (atomicExch(&best_params[0], lock_val) != lock_val) {
                // Spin-wait until we acquire the lock
            }

            // We have the lock, update the parameters
            for (int i = 0; i < num_params; ++i) {
                best_params[i] = params[i];
            }

            // Release the lock by writing back the first parameter
            best_params[0] = params[0];

            // Ensure that the parameter update is visible to all threads
            __threadfence();
        }
    }
};
