#include <iostream>
#include <vector>
#include <random>
#include <set>

#include "utils.h"
typedef std::vector<float> point;

int main()
{
    std::cout << "ok" << std::endl;
    unsigned int num_points = 120; 
    unsigned int num_clusters = 10;
    int bounding_box_min = -5;
    int bounding_box_max =  5;
    unsigned int max_number_of_lloyd_iterations = 100;

    std::vector<point> points;
    points.reserve(num_points);

    std::vector<point> centers;
    centers.reserve(num_clusters);

    std::vector<uint> assignment(num_points, 0);

    kmeans(num_points, num_clusters, bounding_box_min, bounding_box_max, max_number_of_lloyd_iterations, points, centers, assignment);
    return 0;
}