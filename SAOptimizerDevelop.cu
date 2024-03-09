#include <chrono>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <cfloat>
#include "SAOptimizer.cuh"

typedef float NumericType;

template<typename NumericType>
struct SimpleQuadraticObjectiveFunction {
    size_t num_params;

    // Constructor to initialize the number of parameters
    SimpleQuadraticObjectiveFunction(size_t num_params) : num_params(num_params) {}

    __device__
        NumericType operator()(NumericType* params) {
        NumericType sum = 0;
        for (size_t i = 0; i < num_params; ++i) {
            sum += params[i] * params[i]; // Square each parameter and add to sum
        }
        return sum * sum;
    }
};


struct MinIndexOp {
    __host__ __device__
    thrust::tuple<size_t, NumericType> operator()(const thrust::tuple<size_t, NumericType>& a,
                         const thrust::tuple<size_t, NumericType>& b) const {
        return thrust::get<1>(a) < thrust::get<1>(b) ? a : b;
    }
};


__global__ void updateVectorsWithBestParams(float* vectors, const float* bestParams, size_t numVectors, size_t numParams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numVectors * numParams) {
        return;
    }
    int paramIdx = idx % numParams;
    vectors[idx] = bestParams[paramIdx];
}

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

            // Creating a device vector to store the best params per iteration
            thrust::device_vector<NumericType> d_best_params(m_num_params, 0.0f);

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
                optimizer.optimize(256);

                cudaDeviceSynchronize();

                // Create a sequence of indices
                thrust::device_vector<size_t> d_indices(d_objective_values.size());
                thrust::sequence(d_indices.begin(), d_indices.end());

                auto min_tuple = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(d_indices.begin(), d_objective_values.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(d_indices.end(), d_objective_values.end())),
                                thrust::make_tuple((size_t)0, FLT_MAX),
                                MinIndexOp());
                size_t min_index = thrust::get<0>(min_tuple);
                float min_value = thrust::get<1>(min_tuple);

                // Update the best parameters and value
                if (min_value < best_value) {
                    best_value = min_value;
                    thrust::copy_n(d_vectors.begin() + min_index * m_num_params, m_num_params, d_best_params.begin());
                }

                // Update d_vectors with the current best parameters
                int blockSize = 256;
                int numBlocks = (m_num_vectors * m_num_params + blockSize - 1) / blockSize;
                updateVectorsWithBestParams<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_vectors.data()),
                    thrust::raw_pointer_cast(d_best_params.data()),
                    m_num_vectors, m_num_params);

                temperature *= m_cooling_rate;
            }

            // Finally transfer the best params to the cpu
            h_best_params = d_best_params;
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
    size_t num_threads = 1024 * 64;
    NumericType temperature = 1000.0f;
    size_t max_iterations = 10;
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
    std::cout << "Time taken for " << num_params * num_threads * max_iterations << " operations : " << duration << " milliseconds" << std::endl;

    return 0;
}
