#include <iostream>
#include <vector>
#include <random>
#include <set>

typedef std::vector<float> point;

void initialize_points(std::vector<point> & points, const uint num_points, const int range_min, const int range_max)
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

void initialize_centers(const std::vector<point> & points, std::vector<point> & centers, const uint num_clusters)
{
    // initialize maps 

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> uniformDistribution_idx(0, points.size());

    // assign cluster centers
    uint idx;
    for(uint i = 1; i <= num_clusters; i++)
    {
        idx = uniformDistribution_idx(rng);
        centers.push_back(points[idx]);
    }
}

float compute_distance(const point &p1, const point &p2)
{
    float d2 = 0;
    for(uint i = 0; i < p2.size(); i++)
    {
        d2+= std::pow((p2[i]-p1[i]),2);
    }

    return std::sqrt(d2);
}

void assign_points_to_clusters(const std::vector<point> & points, const std::vector<point> & centers,
 std::vector<uint> &assignment, const uint num_clusters)
{
    float min_dist, dist;
    uint idx_assign = 0;
    for(uint p_idx = 0; p_idx < points.size(); p_idx++)
    {
        min_dist = 10000;
        for(uint center_idx = 0; center_idx < num_clusters; center_idx++)
        {
            dist = compute_distance(points[p_idx], centers[center_idx]);
            if (dist < min_dist)
            {
                min_dist = dist;
                idx_assign = center_idx;
            } 
        }
        assignment[p_idx] = idx_assign; //cluster index that corresponds to the min element of d
    }
}

void compute_centroids(const std::vector<point> &points, std::vector<point> centroids, const uint num_clusters)
{

}

void kmeans(const uint num_points, const uint num_clusters, const int range_min, const int range_max, 
uint max_iter, std::vector<point> points, std::vector<point> centers,std::vector<uint> assignment)
{
        initialize_points(points, num_points, range_min, range_max);
        initialize_centers(points,centers,num_clusters);
        assign_points_to_clusters(points, centers, assignment, num_clusters);
        for(uint iter = 0; iter < max_iter; iter ++)
        {
            int i =1;
            // recompute centroids
            // reassign_points: can reuse assign_points_to_clusters
        }
}
int main()
{
    std::cout << "ok here we go" << std::endl;
    unsigned int num_points = 15; 
    unsigned int num_clusters = 3;
    int bounding_box_min = -5;
    int bounding_box_max =  5;
    unsigned int max_number_of_lloyd_iterations = 10;

    std::vector<point> points;
    points.reserve(num_points);

    std::vector<point> centers;
    centers.reserve(num_clusters);

    std::vector<uint> assignment;
    assignment.reserve(num_points);

    kmeans(num_points, num_clusters, bounding_box_min, bounding_box_max, max_number_of_lloyd_iterations, points, centers, assignment);
    // initialize random unif points, clusters
    // send to kernel to compute voronoi cells
    // construct new cluster centers
    // iterate

    return 0;
}