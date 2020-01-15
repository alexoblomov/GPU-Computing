#pragma once

typedef std::vector<float> point;

void initialize_points(std::vector<point> & points, const uint num_points, const int range_min, const int range_max);
void initialize_centers(const std::vector<point> & points, std::vector<point> & centers, const uint num_clusters);
float compute_distance(const point &p1, const point &p2);
void assign_points_to_clusters(const std::vector<point> & points, const std::vector<point> & centers,
 std::vector<uint> &assignment, const uint num_clusters);
 void compute_centroids(const std::vector<point> &points, std::vector<point> centroids, const uint num_clusters);
void kmeans(const uint num_points, const uint num_clusters, const int range_min, const int range_max, 
uint max_iter, std::vector<point> points, std::vector<point> centers,std::vector<uint> assignment);

void create_snapshot(std::vector<point> points, std::vector<point> centroids, 
std::vector<uint> assignment, uint num_points, uint num_clusters, uint iter);