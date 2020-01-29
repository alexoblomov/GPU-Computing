#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <fstream>
#include <iterator>
#include <cassert>
#include <set>

#include "utils.h"

void initialize_points(std::vector<point> & points, uint dim, uint num_points, int range_min, int range_max)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    std::uniform_real_distribution<float> uniformRealDistribution(range_min, range_max);
    float coord;
    
    for(uint i = 0; i < num_points; i++)
    {
        point p;
        for(uint j = 0; j < dim; j++)
        {
            coord = uniformRealDistribution(rng);
            p.emplace_back(coord);
        }      
        points.emplace_back(p);
    }
}

void initialize_centers(const std::vector<point> & points, std::vector<point> & centers, uint num_clusters)
{
    // initialize maps 

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> uniformDistribution_idx(0, points.size()-1);

    // assign cluster centers
    uint idx;
    std::set<int> cluster_indices;

    while (cluster_indices.size() < num_clusters)
    {
        idx = uniformDistribution_idx(rng);
        cluster_indices.insert(idx);
    }

    for (auto idx : cluster_indices) centers.push_back(points[idx]);
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
 std::vector<uint> &assignment, uint num_clusters)
{
    float min_dist, dist;
    uint idx_assign;
    for(uint p_idx = 0; p_idx < points.size(); p_idx++)
    {
        idx_assign = 0;
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
 std::vector<uint> &assignment, uint num_clusters)
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

void compute_centroids(const std::vector<point> &points, uint dim, std::vector<point> &centroids, 
const std::vector<uint> &assignment, uint num_clusters)
{
    std::vector<uint> num_points_per_cluster(num_clusters, 0);

    uint cluster_idx = 0;
    // clear elements in centroid
    for (uint p_idx = 0; p_idx < points.size(); p_idx ++)
    {
        cluster_idx = assignment[p_idx];
        for(uint coord = 0; coord < dim; coord ++)
        {
            centroids[cluster_idx][coord] =  0; 
        }       
    }
    // compute sum of points in a cluster
    for (uint p_idx = 0; p_idx < points.size(); p_idx ++)
    {
        cluster_idx = assignment[p_idx];
        for(uint coord = 0; coord < dim; coord ++)
            centroids[cluster_idx][coord] =  centroids[cluster_idx][coord] + points[p_idx][coord];
        num_points_per_cluster[cluster_idx] = num_points_per_cluster[cluster_idx] + 1;
    }

    // average out this value by the number of points
    for (uint cluster_idx = 0; cluster_idx < num_clusters; cluster_idx ++)
    { 
        for (uint coord = 0; coord < dim; coord ++)
        {
            assert(num_points_per_cluster[cluster_idx] > 0);
            centroids[cluster_idx][coord] = centroids[cluster_idx][coord] /(num_points_per_cluster[cluster_idx]);
        }     
    }
}

void kmeans_iterations(uint num_points, uint dim, uint num_clusters, int range_min, int range_max, 
uint max_iter, std::vector<point> &points, std::vector<point> &centers, std::vector<uint> &assignment)
{
    for(uint iter = 0; iter < max_iter; iter ++)
    {
            compute_centroids(points, dim, centers, assignment, num_clusters);
            if(iter < max_iter-1)
                reassign_points_to_clusters(points, centers, assignment, num_clusters);
    }
}

void kmeans(uint num_points, uint dim, uint num_clusters, int range_min, int range_max, 
uint max_iter, std::vector<point> &points, std::vector<point> &centers, std::vector<uint> &assignment)
{
        initialize_points(points, dim, num_points, range_min, range_max);
        initialize_centers(points,centers,num_clusters);
        assign_points_to_clusters(points, centers, assignment, num_clusters);
        create_snapshot(points,centers,assignment, num_points, num_clusters,0);
        for(uint iter = 0; iter < max_iter; iter ++)
        {
            compute_centroids(points, dim, centers, assignment, num_clusters);
            if(iter < max_iter-1)
                reassign_points_to_clusters(points, centers, assignment, num_clusters);
            //if (iter %20 == 0)
            if (iter > 0)
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