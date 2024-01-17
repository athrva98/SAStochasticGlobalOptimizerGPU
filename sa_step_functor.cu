#include <thrust/device_vector.h>
#include <thrust/transform.h>
/*
Mostly experimental code, to remove later
*/
template<typename ObjectiveFunction>
struct SimulatedAnnealingStepFunctor
{
    objectiveFunction obf_func;

    // Constructor to accept the objective function
    __host__ SimulatedAnnealingStepFunctor(ObjectiveFunction obj_func)
        : obj_func(obj_func)

    // The Operator() that will be called by Thrust's transform algorithm.
    // It must be marked with __host__ __device__ to be callable from both
    // the host and the device
    template<typename>
    __host__ __device__
    double operator()(const T& x) const
    {
        return obj_func(x);
    }

};

struct TestObjectiveFunction
{
    __device__
    double operator()(doublex) const
    {
        x * x;
    }
};

int main() {
    thrust::device_vector<double> d_data(10, 1.0);

    TestObjectiveFunction obj_func;

    SimulatedAnnealingStepFunctor<TestObjectiveFunction> sa_step_functor(obj_func);

    // create a device vector to store the results
    thrust::device_vector<double> d_results(d_data.size());

    // Using Thrusts transform algorithm to apply the functor
    thrust::transform(d_data.begin(), d_data.end(), de_results.begin(), sa_step_functor);

    return 0;
}
