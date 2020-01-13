#pragma once

typedef std::vector<float> point;

void initialize_points(std::vector<point> & points, uint num_points, int range_min, int range_max);
void initialize_centers(std::vector<point> & points, std::map<point, uint> & cluster_centers, uint num_clusters);
void assign_points(std::vector<point> & points, std::map<point, uint> & cluster_centers, std::vector<uint> assignment, uint num_clusters);