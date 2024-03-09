# Simulated Annealing Optimizer

## Overview

The Simulated Annealing Optimizer is a computational algorithm designed to approximate the global optimum of a given function. It draws inspiration from the physical process of heating and then slowly cooling a material to achieve a state of minimal thermal energy. This project implements the simulated annealing technique to identify optimal solutions in complex search spaces. This project mainly uses `thrust` for gpu acceleration. 

## Architecture

The optimization algorithm is structured around a hierarchical iteration framework, incorporating both outer and inner iterations to progressively refine the solution. At the heart of this framework lies the concept of sub-iterations, which distinguishes between the broader, overarching iterations (outer) and the more granular, detailed computational steps (inner).

For each cycle of the outer iteration, a new instance of the simulated annealing algorithm is instantiated. This instance is responsible for launching a multitude of parallel threads on the GPU, each thread dedicated to exploring the solution space and striving to minimize the objective function, often referred to as the loss in optimization terminology.

The essence of this iterative process lies in its ability to leverage the solutions derived from the concurrent simulated annealing executions. Specifically, the solution obtained at the conclusion of an outer iteration serves as the starting point or initial guess for the subsequent outer iteration. This strategy not only ensures that the search for the optimal solution is continuous but also allows for the progressive refinement and convergence towards the global minimum.

By utilizing the GPU's parallel computing capabilities, the algorithm efficiently explores a vast solution space, significantly accelerating the optimization process. This hierarchical iteration approach, combined with the power of parallel computing, forms the foundation of the algorithm's architecture, enabling it to tackle complex optimization problems with enhanced efficacy and speed.

## Features

- **Algorithm Foundation**: Utilizes the simulated annealing methodology for optimization, simulating the process of annealing in metallurgy.
- **Customizability**: Offers flexibility to adapt to various objective functions and optimization scenarios.
- **GPU Acceleration**: Leverages CUDA for efficient computation, significantly reducing execution time on compatible NVIDIA GPUs.

## Current Development Status

The project is currently in the development phase and supports compilation and execution on Windows platforms. It is in an early stage, with ongoing testing and refinement. Users may encounter limitations or unresolved issues.

## Compilation Instructions

### Prerequisites

- NVIDIA CUDA Toolkit (Version 12.2 recommended)
- Microsoft Visual Studio (2022 or later recommended)
- Windows 10 or newer, with appropriate SDK

### Environment Setup

1. **CUDA Flags and Windows SDK Directory**:
   Configure the CUDA flags to include the path to your Windows SDK directory.
   Example:
   
   ```shell
   set CUDAFE_FLAGS=--sdk_dir "C:\Program Files (x86)\Windows Kits\10\"
   ```

2. **Identify Essential Paths**:
   
   - CUDA Toolkit Path: e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2`
   - Microsoft Visual Studio Compiler Path: e.g., `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\HostX64\x64`

### Compiling CUDA Files

Utilize the NVIDIA CUDA Compiler (NVCC) in a Command Prompt window with the following structure:

```shell
"<Path_to_CUDA_Toolkit>/bin/nvcc.exe" --use-local-env -ccbin "<Path_to_MSVC_Compiler>" -x cu --keep-dir x64\Release -maxrregcount=0 --machine 64 --compile -cudart static -o "<Output_Object_File>" "<Input_CU_File>"
```

## Usage

### Defining an Objective Function

The objective function represents the problem you're trying to solve, with the optimizer aiming to find the input values that minimize the output of this function.

Objective functions should be defined with the following considerations:

1. **Input**: A pointer to an array of `NumericType`, representing the parameters being optimized.
2. **Output**: A single `NumericType` value, representing the "cost" or "fitness" associated with the input parameters. The optimizer seeks to minimize this value.
3. **Compatibility**: The function must be __device__ callable, meaning it can be executed on the GPU.

note: `NumericType` can be defined as a `typedef` see, the code for a better understanding.

### Example: Quadratic Objective Function

Below is an example of a simple quadratic objective function, which is compatible with the optimizer. 

```cpp
template<typename NumericType>
class QuadraticObjectiveFunction {
public:
    __device__
    NumericType operator()(const NumericType* params, size_t num_params) {
        NumericType sum = 0;
        for (size_t i = 0; i < num_params; ++i) {
            sum += params[i] * params[i];
        }
        return sum;
    }
};
```

### Integration with the Optimizer

To use this objective function with the Simulated Annealing Optimizer, follow these steps:

1. **Define the Objective Function Instance**: Create an instance of your objective function. In this case, a `QuadraticObjectiveFunction<NumericType>`.

2. **Initialize the Optimizer**: Pass the objective function instance, along with other required parameters, to the `SimulatedAnnealingOptim` constructor.

3. **Execute the Optimization**: Call the `minimize` method to start the optimization process.

### Example Code Snippet

```cpp
#include "SAOptimizer.cuh" 

// assuming that your objective function is defined in this file, make sure to follow the above pattern for implementing the objective function 

int main() {
    // Define optimization parameters
    const size_t num_params = 5; // Number of parameters to optimize
    size_t num_vectors = 1024; // Number of solutions to evaluate per iteration
    NumericType temperature = 1000.0f; // Starting temperature
    size_t max_iterations = 1000; // Maximum number of iterations
    int seed = 42; // Random seed for reproducibility
    float cooling_rate = 0.95f; // Rate at which the temperature decreases
    bool warm_start = false; // Indicates whether to start from a previous solution
    float step_size = 0.1f; // Size of the step to take in parameter space

    // Instantiate the objective function
    QuadraticObjectiveFunction<NumericType> obj_func;

    // Initialize the optimizer with the objective function and other parameters
    SimulatedAnnealingOptim<QuadraticObjectiveFunction<NumericType>, NumericType> optimizer(
        obj_func, num_params, num_vectors, temperature, max_iterations,
        seed, cooling_rate, warm_start, step_size);

    // Run the optimization
    optimizer.minimize();

    return 0;
}
```

Users can replace the `QuadraticObjectiveFunction` with their own problem-specific objective functions, following the pattern established here to customize the optimization process to their needs.

## Performance Insights

- **Execution Time** (03/09/2024): Approximately 
  
  ```bash
  Time taken for 1,310,720,000 operations : 3389 milliseconds
  ```

## Contributing

Contributors are welcome to fork the repository, explore the code, and submit pull requests with enhancements, bug fixes, or optimizations. Collaboration is encouraged to refine and expand the capabilities of the optimizer.

## Disclaimer

This software is provided as-is, without warranty. Users should proceed with caution, understanding that the project is a work in progress and may contain bugs or inaccuracies.

## Acknowledgements

The development of this project is supported by contributions from the community and insights from existing research on simulated annealing and optimization techniques.
