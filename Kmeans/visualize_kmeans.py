# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:17:37 2020

@author: kennarda
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def color_points(points, assignment, colors):
    for i in range(len(assignment)):
        plt.scatter(points[i,0], points[i,1], c=colors[int(assignment[i])])
   

def color_centroids(centroids, assignment, colors):
    ncolor = 0
    for i in range(len(centroids)):
        plt.scatter(centroids[i,0], centroids[i,1],  marker = '^', c = colors[ncolor])
        ncolor = ncolor + 1
        
num_iter = 100
num_points = 120
num_clusters = 10
save_every = 20

fname = "build/points_n" + str(num_points) + "_iter" + str(0) + ".txt"
points = np.loadtxt(fname)

fig = plt.figure()
plt.scatter(points[:,0], points[:,1])

#colors = ['b', 'c', 'k', 'y', 'g']
colors = cm.rainbow(np.linspace(0, 1, num_clusters))

for i in range(num_iter):
    if i% save_every == 0:
        fname = "build/assignment_n" + str(num_points) + "_iter" + str(i) + ".txt"
        assignment = np.loadtxt(fname)
        color_points(points, assignment, colors)
        
        fname = "build/centroids_n" + str(num_points) + "_iter" + str(i) + ".txt"
        centroids = np.loadtxt(fname)
        #color_centroids(centroids, assignment, colors)
        plt.show()
    
