/*
 **  PROGRAM: Approximation of pi
 **
 **  PURPOSE: This program will numerically compute the integral of
 **           4/(1+x*x)
 **
 **           from 0 to 1. The value of this integral is pi.
 **           The is the original sequential program. It uses the timer
 **           from the OpenMP runtime library
 **
 **  USAGE: ./pi
 **
 */

#include "util.hpp"
#include <err_code.h>
#include "device_picker.hpp"
#include <iostream>

static long num_steps = 100000000;
double step;
extern double wtime();   // returns time since some fixed past point (wtime.c)

int main ()
{
    /* Sequential Code */
    int i;
    double x, pi, sum = 0.0;
    step = 1.0/(double) num_steps;

    util::Timer timer;

    for (i=1;i<= num_steps; i++){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;
    double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    std::cout<<"pi with "<<num_steps<<" steps is "
        << pi <<" in "
        <<run_time<<" seconds"<<std::endl;

    /* Parallel Code */
    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex);

    // Get list of devices
    std::vector<cl::Device> devices;
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices)
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
     timer.reset();

    // Load in kernel source, creating a program object for the context
    cl::Program program(context, util::loadProgram("pi.cl"), true);
    

    
}

