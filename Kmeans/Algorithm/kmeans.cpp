#include <iostream>
#include <vector>
#include <random>


typedef std::vector<float> point;

void initialize_points(std::vector<point> & points, uint num_points, int range_min, int range_max)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    std::uniform_real_distribution<float> uniformRealDistribution_x(range_min, range_max);
    std::uniform_real_distribution<float> uniformRealDistribution_y(range_min, range_max);

    float x, y;
    for(uint i = 0; i < num_points; i++)
    {
        x = uniformRealDistribution_x(rng);
        y = uniformRealDistribution_y(rng);
        points.emplace_back(point{x,y});
    }
}

void initialize_centers(std::vector<point> & points, std::vector<point> & cluster_centers, uint num_clusters)
{

    std::random_device dev;
    std::mt19937 rng(dev());

    std::uniform_int_distribution<int> uniformDistribution_idx(0, points.size());

    uint idx;
    for(uint i = 0; i < num_clusters; i++)
    {
        idx = uniformDistribution_idx(rng);
        // get point index belongs to ? use std::set ?
        // store point in cluster
    }
}


int main()
{
    std::cout << "ok here we go" << std::endl;
    unsigned int num_points = 100; 
    unsigned int num_clusters = 10;
    int range_min = -5;
    int range_max =  5;

    std::vector<point> points;
    initialize_points(points, num_points, range_min, range_max);

    std::vector<point> cluster_centers;

    // initialize random unif points, clusters
    // send to kernel to compute voronoi cells
    // construct new cluster centers
    // iterate

    return 0;
}