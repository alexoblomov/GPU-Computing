#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <cassert>
#include "utils.h"

typedef std::vector<float> point;

int main()
{
    std::cout << "ok" << std::endl;
    unsigned int num_points = 4; 
    unsigned int point_dimension = 2;
    unsigned int num_clusters = 3;
    int bounding_box_min = -5;
    int bounding_box_max =  5;
    unsigned int max_number_of_lloyd_iterations = 2;

    assert(num_points >= num_clusters);
    std::vector<point> points;
    //points.reserve(num_points);

    std::vector<point> centers;
    //centers.reserve(num_clusters);

    std::vector<uint> assignment(num_points, 0);

    kmeans(num_points,point_dimension, num_clusters, bounding_box_min, bounding_box_max, max_number_of_lloyd_iterations, points, centers, assignment);
    return 0;
}