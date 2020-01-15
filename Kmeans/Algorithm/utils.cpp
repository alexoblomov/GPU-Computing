#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <fstream>
#include <iterator>

#include "utils.h"

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

void reassign_points_to_clusters(const std::vector<point> & points, const std::vector<point> & centers,
 std::vector<uint> &assignment, const uint num_clusters)
{
    float cur_dist, dist;
    uint idx_assign;
    for(uint p_idx = 0; p_idx < points.size(); p_idx++)
    {
        cur_dist = compute_distance(points[p_idx], centers[assignment[p_idx]]);
        idx_assign = assignment[p_idx];
        for(uint center_idx = 0; center_idx < num_clusters; center_idx++)
        {
            dist = compute_distance(points[p_idx], centers[center_idx]);
            if (dist < cur_dist)
            {
                cur_dist = dist;
                idx_assign = center_idx;
            } 
        }
        assignment[p_idx] = idx_assign; //cluster index that corresponds to the min element of d
    }
}

void compute_centroids(std::vector<point> &points, std::vector<point> centroids, 
std::vector<uint> assignment, uint num_clusters)
{
    std::vector<point> means(num_clusters, point{0,0});
    //uint num_points_per_cluster[num_clusters] = {0};
    std::vector<uint> num_points_per_cluster(num_clusters, 0);
    uint cluster_idx = 0;
    for (uint p_idx = 0; p_idx < points.size(); p_idx ++)
    {
        cluster_idx = assignment[p_idx];
        for(uint coord = 0; coord < points[p_idx].size(); coord ++)
        {
            means[cluster_idx][coord] =  means[cluster_idx][coord] + points[p_idx][coord]; //
        }
        num_points_per_cluster[cluster_idx] ++;
    }
    for (uint cluster_idx = 0; cluster_idx < num_clusters; cluster_idx ++)
    { // avoid 0 division
        for (uint coord = 0; coord < points[0].size(); coord ++)
            means[cluster_idx][coord] = means[cluster_idx][coord] /(num_points_per_cluster[cluster_idx]+ 0.01);
    }
}

void kmeans(const uint num_points, const uint num_clusters, const int range_min, const int range_max, 
uint max_iter, std::vector<point> points, std::vector<point> centers,std::vector<uint> assignment)
{
        initialize_points(points, num_points, range_min, range_max);
        initialize_centers(points,centers,num_clusters);
        assign_points_to_clusters(points, centers, assignment, num_clusters);
        for(uint iter = 0; iter < max_iter; iter ++)
        {
            compute_centroids(points, centers, assignment, num_clusters);
            reassign_points_to_clusters(points, centers, assignment, num_clusters);
            if (iter %20 == 0)
                create_snapshot(points,centers,assignment, num_points, num_clusters,iter);
        }
}

void write_vector_of_vector_to_file(std::ofstream &stream, std::vector<point> &points)
{
    for (const auto &vt : points)
    {
        std::copy(vt.cbegin(), vt.cend(),
                  std::ostream_iterator<float>(stream, " "));
        stream << '\n';
    }
}

void write_vector_to_file(std::ofstream &stream, std::vector<uint> &integers)
{
    for (const auto & elem : integers) stream << elem << '\n';
}

void create_snapshot(std::vector<point> points, std::vector<point> centroids, 
std::vector<uint> assignment, uint num_points, uint num_clusters, uint iter)
{
    std::string assignment_file = "assignment_n" + std::to_string(num_points) + 
    "_iter" + std::to_string(iter) + ".txt";
    std::ofstream assignment_fstream(assignment_file);
    write_vector_to_file(assignment_fstream, assignment);

    std::string points_file = "points_n" + std::to_string(num_points) + 
    "_iter" + std::to_string(iter) + ".txt";
    std::ofstream points_fstream(points_file);
    write_vector_of_vector_to_file(points_fstream, points);

    std::string centroids_file = "centroids_n" + std::to_string(num_points) + 
    "_iter" + std::to_string(iter) + ".txt";
    std::ofstream centroids_fstream(centroids_file);
    write_vector_of_vector_to_file(centroids_fstream, centroids);
}