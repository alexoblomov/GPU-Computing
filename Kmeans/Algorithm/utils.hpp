#include <iostream>
#include <vector>
#include <random>
#include <map>

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

void initialize_centers(std::vector<point> & points, std::map<point, uint> & cluster_centers, uint num_clusters)
{
    // initialize map 
    for(uint i = 0; i < points.size(); i++)
    {
        cluster_centers[points[i]] = 0;
    }

    for(auto elem : cluster_centers)
    {
        for(auto point : elem.first)
            std::cout << point;
        std::cout << "," << elem.second << std::endl;
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> uniformDistribution_idx(0, points.size());

    // assign cluster centers
    uint idx;
    for(uint i = 1; i <= num_clusters; i++)
    {
        idx = uniformDistribution_idx(rng);
        // find idx of point in cluster center
        cluster_centers[points[idx]] = i;
    }
    for(auto elem : cluster_centers)
    {
        for(auto point : elem.first)
            std::cout << point;
        std::cout << "," << elem.second << std::endl;
    }
}

void assign_points(std::vector<point> & points, std::map<point, uint> & cluster_centers, std::vector<uint> assignment, uint num_clusters)
{
    // float []d;
    // for(p : points)
    // {
    //     for(uint center = 0; center < num_clusters; center ++)
    //     {
    //         d[center] = compute_distance(p, center);
    //         //assignment[p] = cluster index that corresponds to the min element of d
    //     }
        
    // }
}
