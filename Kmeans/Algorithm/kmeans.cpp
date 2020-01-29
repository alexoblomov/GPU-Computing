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

    initialize_points(points, point_dimension, num_points, bounding_box_min, bounding_box_max);
    initialize_centers(points, centers, num_clusters);
    assign_points_to_clusters(points, centers, assignment, num_clusters);

    // initialize vectors for the GPU computations with the same values
    std::vector<point> g_points = points;
    std::vector<point> g_centers = centers;
    std::vector<uint> g_assignment =  assignment;

    // ------------------------------------------------------------------
    // Run sequential kmeans
    // ------------------------------------------------------------------
    timer.reset();
    start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    kmeans_iterations(num_points,point_dimension, num_clusters, bounding_box_min, bounding_box_max, max_iter, points, centers, assignment);
    run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

    std::cout << "Sequential algorithm time: " << run_time-start_time << "s" << std::endl;
    
    // ------------------------------------------------------------------
    // Run parallel kmeans
    // ------------------------------------------------------------------
    timer.reset();


    cl::Buffer d_P, d_C, d_A;
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

        // Setup the buffers
        std::cout << "Setting up the device buffers "<< "\n";
        d_A = cl::Buffer(context, g_assignment.begin(), g_assignment.end(), false);
        std::cout << "Set up the device buffer for A "<< "\n";

        d_P = cl::Buffer(context, points.begin(), points.end(), true);
        d_C = cl::Buffer(context, g_centers.begin(), g_centers.end(), false);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("kmeans.cl"), true);

        // Create the compute kernel from the program
        cl::Kernel kernel_km = cl::Kernel(program, "_kmeans");

        // Display max group size for execution
        std::cout << "\nWork Group Size " << kernel_km.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;
        std::cout << "Work Group Memory size " << kernel_km.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << std::endl;

        // Initialize arguments of kernel
        kernel_km.setArg(0, num_points);
        kernel_km.setArg(1, num_clusters);
        kernel_km.setArg(2, max_iter);
        kernel_km.setArg(3, point_dimension);
        kernel_km.setArg(4, d_P);
        kernel_km.setArg(5, d_C);
        kernel_km.setArg(6, d_A);

        std::cout << "able to send data to kernel" << std::endl;
        // Set workspace and workgroup topologies
        cl::NDRange global(num_clusters, num_clusters);
        cl::NDRange local(16, 16);

        std::cout << "able to set topology" << std::endl;

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