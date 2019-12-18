#define __CL_ENABLE_EXCEPTIONS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <iostream>
#include <string>
#include <cmath>

#include "oclhelper.h"

#define WG_SIZE 256

void verifyResult(cl_float result, cl_float validResult)
{
  if (std::fabs(result - validResult) < 1.e-5)
    std::cout << "PASSED!" << std::endl;
  else
    std::cout << "FAILED (" << result << "!=" << validResult << ")!" << std::endl;
}

void outputInfo(float result, double elapsedTime, int workitems)
{
  std::cout << "Output: " << result << " / Time: " << elapsedTime << " msecs (" << (workitems / elapsedTime / 1000000.0) << " billion elements/second)" << std::endl;
}

int main(void)
{
  std::cout << std::string(15, '=') << " Reduction sum optimization benchmark" << std::endl;
  oclHelper::oclRuntime runtime(0, 0);

  runtime.displayPlatformInfo();
  runtime.displayDeviceInfo();
  runtime.addCompileOptions("-DWG_SIZE=" + std::to_string(WG_SIZE));

  cl::Program program;
  runtime.buildProgramFromFile(program, "reduction_sum.cl");
  cl::CommandQueue queue = runtime.getCmdQueue();

  float validResult = std::log(2);

  std::vector<std::string> kernels = {"reduce1", "reduce2", "reduce3", "reduce4", "reduce5", "reduce6", "reduce7", "reduceMdb"};
  for (auto ker : kernels)
  {
    // Create kernel and set NDRange size
    {
      double elapsedTime = 0;
      const uint TOTAL_ITS = 10;
      uint factor = 1;
      if (!ker.compare(std::string("reduce7")) || !ker.compare(std::string("reduce4")))
      {
        factor = 2;
      }
      cl::Kernel kernel(program, ker.c_str());

      // Get maximum of compute unit for given kernel
      int MAX_CUs = runtime.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

      cl::NDRange localSize(WG_SIZE);
      uint size = 2 * MAX_CUs * 1024 * WG_SIZE;
      size_t nb_wg = size / WG_SIZE / factor;

      std::cout << "\n"
                << std::string(10, '=') << "\t" << ker << std::endl;
      std::cout << "Launching NDRange size of " << nb_wg << " workgroups with " << WG_SIZE << " workitems per workgroup" << std::endl;

      // Get number of workgroups
      cl::Buffer d_i(runtime.getContext(), CL_MEM_READ_WRITE, size * sizeof(cl_float));
      cl::Buffer d_o(runtime.getContext(), CL_MEM_READ_WRITE, nb_wg * sizeof(cl_float));

      cl::Event eKernel;

      for (int k = 0; k < TOTAL_ITS; k++)
      {
        nb_wg = size / WG_SIZE / factor;
        cl_float *map_i = (cl_float *)queue.enqueueMapBuffer(d_i, CL_TRUE, CL_MAP_WRITE, 0, size * sizeof(float));

        /// Initialize input data at each iteration
        for (int i = 0; i < size; i++)
        {
          map_i[i] = std::pow(-1, i) * 1 / (i + 1);
        }
        queue.enqueueUnmapMemObject(d_i, map_i);

        ///  First stage reduction : reduction of data into workgroup
        {
          cl::NDRange gridSize(size / factor);
          kernel.setArg(0, d_i);
          kernel.setArg(1, d_o);
          kernel.setArg(2, localSize[0] * sizeof(cl_float), NULL);
          kernel.setArg(3, size);

          // Shared memory kernel
          cl::Event eKernel;

          queue.enqueueNDRangeKernel(kernel, cl::NullRange, gridSize, localSize, NULL, &eKernel);
          queue.finish();

          auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
          auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
          elapsedTime += (infFinish - infStart) / 1000000.;
        }

        /// Second stage reduction : reduction of workgroup
        {
          // Grid size is half the number of workgroup from stage 1
          cl::NDRange gridSize(nb_wg / factor);

          // Swap input and ouput from previous stage
          kernel.setArg(0, d_o);
          kernel.setArg(1, d_i);
          kernel.setArg(2, localSize[0] * sizeof(cl_float), NULL);
          kernel.setArg(3, nb_wg);

          queue.enqueueNDRangeKernel(kernel, cl::NullRange, gridSize, localSize, NULL, &eKernel);
          queue.finish();

          auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
          auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
          elapsedTime += (infFinish - infStart) / 1000000.;

          nb_wg = gridSize[0] / WG_SIZE;
        }
      }
      /// Third stage reduction : finalize reduction on CPU
      cl_float *map = (cl_float *)queue.enqueueMapBuffer(d_i, CL_TRUE, CL_MAP_READ, 0, nb_wg * sizeof(cl_float));
      cl_float result = 0;
      for (int i = 0; i < nb_wg; i++)
      {
        result += map[i];
      }
      queue.enqueueUnmapMemObject(d_o, map);
      elapsedTime /= TOTAL_ITS;

      // Verify result
      outputInfo(result, elapsedTime, size);
      verifyResult(result, validResult);
    }
  }
  return 0;
}