# Simulated Annealing Optimizer (Work in Progress)

Welcome to the repository of the most chill optimization algorithm out there, the Simulated Annealing Optimizer! It's like taking the concept of "cooling down" quite literally. 😎❄️

# What's Cooking? 🍳

This optimizer is based on the concept of Simulated Annealing, where we metaphorically heat up a system and then slowly cool it down to find a state of minimum energy, or in our case, the optimal solution. It's like trying to find the comfiest spot on your bed, but mathematically!

# Current Status 🚧

Currently, the optimizer can be built and run on windows. Note that it is not tested extensively yet, so there may be some issues.

# Compiling CUDA Files with NVCC

This guide provides instructions on how to compile CUDA .cu source files using the NVIDIA CUDA Compiler (NVCC) on a Windows system. Please ensure that you have the CUDA Toolkit and Microsoft Visual Studio installed before proceeding.

Prerequisites

- NVIDIA CUDA Toolkit (e.g., CUDA v12.2)
- Microsoft Visual Studio (e.g., Visual Studio 2022)
- Windows 10 and 10+ SDK

Setting Up Your Environment

- Set CUDA Flags and Windows SDK Directory

- Before running the NVCC command, set the CUDA flags with the path to your Windows SDK directory. Replace `<Windows_SDK_Dir>` with your Windows SDK path.


```
set CUDAFE_FLAGS=--sdk_dir "<Windows_SDK_Dir>"
```

Example:
```
    set CUDAFE_FLAGS=--sdk_dir "C:\Program Files (x86)\Windows Kits\10\"
```

Locate Required Paths

- CUDA Toolkit Path: Find the path where CUDA Toolkit is installed.

    Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2

- Microsoft Visual Studio Compiler Path: Locate the MSVC compiler path in your Visual Studio installation.

    Example: C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\HostX64\x64

# Compiling CUDA Files

- Open Command Prompt

- Open a Command Prompt window where you'll run the NVCC compilation command.

- Run NVCC Command

- Use the following command structure to compile your .cu files. Replace the placeholders with the actual paths and file names.

```
"<Path_to_CUDA_Toolkit>/bin/nvcc.exe" --use-local-env -ccbin "<Path_to_MSVC_Compiler>" -x cu --keep-dir x64\Release -maxrregcount=0 --machine 64 --compile -cudart static -o "<Output_Object_File>" "<Input_CU_File>"
```

Example:

```
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe" --use-local-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\HostX64\x64" -x cu --keep-dir x64\Release -maxrregcount=0 --machine 64 --compile -cudart static -o "C:\YourProject\x64\Release\YourOutputFile.obj" "C:\YourProject\YourSourceFile.cu"
```
- `<Path_to_CUDA_Toolkit>`: Path to your CUDA Toolkit installation.
- `<Path_to_MSVC_Compiler>`: Path to your MSVC compiler in Visual Studio.
- `<Output_Object_File>`: Desired path and name of the compiled object file.
- `<Input_CU_File>`: Path and name of the CUDA .cu source file you want to compile.

## Additional Notes

- The NVCC command provided is configured for Windows systems with Visual Studio. Adjustments might be needed for different environments or CUDA versions.


# Contributions 🤝

Feel free to fork this project, play around with the code, and submit pull requests. It's like a potluck; bring your own spices and let's make this dish even more delicious!

# Disclaimer ⚠️

This code is not for the faint of heart. It's a work in progress, and like a wild roller coaster, it has its ups and downs. Use it at your own risk, and remember, with great power comes great responsibility (to debug).

# Stay Cool 😎

Remember, this optimizer is all about cooling down, so take a deep breath, grab a cup of coffee, and enjoy the process. Happy optimizing!
