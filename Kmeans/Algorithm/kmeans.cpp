#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <cassert>
#include "utils.h"

// Parallel imports
#define __CL_ENABLE_EXCEPTIONS

#include "util.hpp"
#include "cl.hpp"
#include <err_code.h>
#include "device_picker.hpp"


typedef std::vector<float> point;

int main(int argc, char *argv[])
{
    double start_time; // Starting time
    double run_time;   // Timing
    util::Timer timer; // Timing

    // MODEL PARAMETERS
    unsigned int num_points = 15; 
    unsigned int point_dimension = 2;
    unsigned int num_clusters = 3;
    int bounding_box_min = -5;
    int bounding_box_max =  5;
    unsigned int max_iter = 5;

    assert(num_points >= num_clusters);

    std::vector<point> points;
    std::vector<point> centers;
    std::vector<uint> assignment(num_points, 0);

    // ------------------------------------------------------------------
    // Run sequential kmeans
    // ------------------------------------------------------------------
    timer.reset();
    start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    kmeans(num_points,point_dimension, num_clusters, bounding_box_min, bounding_box_max, max_iter, points, centers, assignment);
    run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

    std::cout << "Sequential algorithm time: " << run_time-start_time << "s" << std::endl;
    
    // ------------------------------------------------------------------
    // Run parallel kmeans
    // ------------------------------------------------------------------
    timer.reset();

    std::vector<point> g_points;
    std::vector<point> g_centers;
    std::vector<uint> g_assignment(num_points, 0);
    initialize_points(g_points, point_dimension, num_points, bounding_box_min, bounding_box_max);
    initialize_centers(g_points, g_centers, num_clusters);
    assign_points_to_clusters(g_points, g_centers, g_assignment, num_clusters);

    cl::Buffer d_P, d_C, d_A; // Matrices in device memory
    // write function to initialize d_i
        
    try
    {
        cl_uint deviceIndex = 0;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex > numDevices)
        {
            std::cout << "Invalid device index (try '--list')\n";
            return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("kmeans.cl"), true);

        // Create the compute kernel from the program
        cl::Kernel kernel_km = cl::Kernel(program, "_kmeans");

        // Display max group size for execution
        std::cout << "\nWork Group Size " << kernel_km.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;
        std::cout << "Work Group Memory size " << kernel_km.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << std::endl;

        // Initialize arguments of kernel
        kernel_km.setArg(0, num_points);
        kernel_km.setArg(1, point_dimension);
        kernel_km.setArg(2, num_clusters);
        kernel_km.setArg(3, bounding_box_min);
        kernel_km.setArg(4, bounding_box_max);
        kernel_km.setArg(5, max_iter);
        kernel_km.setArg(6, d_P);
        kernel_km.setArg(7, d_C);
        kernel_km.setArg(8, d_A);

        // Set workspace and workgroup topologies
        cl::NDRange global(num_clusters, num_clusters);
        cl::NDRange local(16, 16);

        queue.enqueueNDRangeKernel(kernel_km, cl::NullRange, global, local);

        queue.finish();
    }
    catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    
    return 0;
}