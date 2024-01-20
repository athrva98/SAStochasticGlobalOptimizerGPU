#include <chrono>
#include "SAOptimizer.cuh"

typedef double NumericType;

template<typename NumericType>
struct SimpleQuadraticObjectiveFunction {
    size_t num_params; // Number of parameters

    // Constructor to initialize the number of parameters
    SimpleQuadraticObjectiveFunction(size_t num_params) : num_params(num_params) {}

    __device__
        NumericType operator()(NumericType* params) {
        NumericType sum = 0;
        for (size_t i = 0; i < num_params; ++i) {
            sum += params[i] * params[i]; // Square each parameter and add to sum
        }
        return abs(sum - 50); // Example operation
    }
};

template<typename ObjectiveFunctionType, typename NumericType>
class SimulatedAnnealingOptim {

public:
    SimulatedAnnealingOptim(ObjectiveFunctionType obj_func,
        size_t num_params,
        size_t num_threads,
        NumericType temperature, size_t max_iterations,
        int seed, float cooling_rate,
        bool warm_start, float step_size) : m_obj_func(obj_func), m_num_params(num_params), m_num_vectors(num_threads),
        m_temperature(temperature), m_max_iterations(max_iterations), m_seed(seed), m_cooling_rate(cooling_rate),
        m_warm_start(warm_start), m_step_size(step_size) {};

        void minimize() {
            // Initialize vectors and objective values
            thrust::host_vector<NumericType> h_vectors(m_num_vectors * m_num_params);
            thrust::host_vector<NumericType> h_objective_values(m_num_vectors, std::numeric_limits<NumericType>::max());

            // Initialize with some values only once, before the loop
            for (size_t i = 0; i < m_num_vectors; ++i) {
                for (size_t j = 0; j < m_num_params; ++j) {
                    h_vectors[i * m_num_params + j] = 0; // Initialize each parameter
                }
            }

            // Copy to device
            thrust::device_vector<NumericType> d_vectors = h_vectors;
            thrust::device_vector<NumericType> d_objective_values = h_objective_values;

            // Iterative optimization approach
            NumericType temperature = m_temperature;
            NumericType best_value = std::numeric_limits<NumericType>::max();
            thrust::host_vector<NumericType> h_best_params(m_num_params, 0);

            for (int iter = 0; iter < m_max_iterations; ++iter) {
                SAOptimizer<ObjectiveFunctionType, NumericType> optimizer(
                    m_obj_func, temperature, m_num_params, m_seed,
                    thrust::raw_pointer_cast(d_vectors.data()),
                    thrust::raw_pointer_cast(d_objective_values.data()), m_num_vectors,
                    m_warm_start, m_step_size);

                // Call the optimize method
                optimizer.optimize(256); // Assuming block size of 256, can be adjusted

                cudaDeviceSynchronize();

                // Update the best parameters and value
                auto min_element_iter = thrust::min_element(d_objective_values.begin(), d_objective_values.end());
                size_t min_index = min_element_iter - d_objective_values.begin();
                if (*min_element_iter < best_value) {
                    best_value = *min_element_iter;
                    thrust::copy_n(d_vectors.begin() + min_index * m_num_params, m_num_params, h_best_params.begin());
                }

                // Fill d_vectors with the current best parameters for the next iteration
                for (size_t i = 0; i < m_num_vectors; ++i) {
                    thrust::copy(h_best_params.begin(), h_best_params.end(), d_vectors.begin() + i * m_num_params);
                }

                temperature *= m_cooling_rate;
            }

            // Output the results
            std::cout << "Optimal value: " << best_value << std::endl;
            std::cout << "Optimal parameters: ";
            for (size_t i = 0; i < m_num_params; ++i) {
                std::cout << h_best_params[i] << " ";
            }
            std::cout << std::endl;
        }




private:
    ObjectiveFunctionType m_obj_func;
    size_t m_num_params;
    size_t m_num_vectors;
    NumericType m_temperature;
    size_t m_max_iterations;
    int m_seed;
    float m_cooling_rate;
    bool m_warm_start;
    float m_step_size;

    // TODO: define host_vectors to hold the solutions
};


int main() {
    const size_t num_params = 2000;
    SimpleQuadraticObjectiveFunction<NumericType> obj_func(num_params);
    size_t num_threads = 1024;
    NumericType temperature = 1000.0f;
    size_t max_iterations = 1000;
    int seed = 1;
    float cooling_rate = 0.95;
    bool warm_start = false;
    float step_size = 1e-03;

    SimulatedAnnealingOptim<SimpleQuadraticObjectiveFunction<NumericType>, NumericType> sa_optimizer(
        obj_func, num_params, num_threads, temperature, max_iterations, seed, cooling_rate, warm_start,
        step_size);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function you want to time
    sa_optimizer.minimize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Output the time taken
    std::cout << "Time taken for " << num_params * num_threads * max_iterations << " evaluations : " << duration << " milliseconds" << std::endl;

    return 0;
}
