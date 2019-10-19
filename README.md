# MACHINE LEARNING IN PYTHON
-------------------------------

## Hierarchical Clustering:
-------------------------------
Hierarchical clustering algorithms build a hierarchy of clusters where each node is a cluster consisting of the clusters of its daughter nodes.
Strategies for hierarchical clustering generally fall into two types: **1. Divisive** and **2. Agglomerative**.

**Divisive** is top-down, so you start with all observations in a large cluster and break it down into smaller pieces.We can think about divisive as "dividing" the cluster.

**Agglomerative** is the opposite of divisive, so it is bottom-up, where each observation starts in its own cluster and pairs of clusters are merged together as they move up the hierarchy.
Agglomeration means to amass or collect things, which is exactly what this does with the cluster.

The Agglomerative approach is more popular among data scientists.

**AGGLOMERATIVE CLUSTERING:**

This method builds the hierarchy from the individual elements by progressively merging
clusters. 

In our example, let’s say we want to cluster 6 cities in India based on their distances from one another.
They are: kolkata,Dhanbad,Allahabad,.
We construct a distance matrix at this stage, where the numbers in the row i column j is
the distance between the i and j cities. In fact, this table shows the distances between
each pair of cities.

|             |**Kolkata**|**Dhanbad**|**Allahabad**|**Delhi**|**Mumbai**|**Chennai**|
|-------------|----------|--------|-------|---------|--------|---------|
|**Kolkata**||277|792|1491|2079|1681|
|**Dhanbad**|||528|1226|2020|1743|
|**Allahabad**||||692|1486|1788|
|**Delhi**|||||1417|2209|
|**Mumbai**||||||1336|
|**Chennai**|||||||


The algorithm is started by assigning each city to its own cluster.So, if we have 6 cities, we have 6 clusters, each containing just one city.
The first step is to determine which cities -- let’s call them clusters from now on
-- to merge into a cluster. Usually, we want to take the two closest clusters according
to the chosen distance. Looking at the distance matrix, Kolkata and
Dhanbad are the closest clusters. So, we make a cluster out of them.

|             |**Kolkata**/**Dhanbad**|**Allahabad**|**Delhi**|**Mumbai**|**Chennai**|
|-------------|------------------|-------|---------|--------|---------|
|**Kolkata**/**Dhanbad**||780|1465|2074|1675|
|**Allahabad**|||692|1486|1788|
|**Delhi**||||1417|2209|
|**Mumbai**|||||1336|
|**Chennai**||||||

Please notice that we just use a simple 1-dimentional distance feature here, but our object can
be multi-dimensional, and distance measurement can be either Euclidean, Pearson, average
distance, or many others, depending on data type and domain knowledge.
Anyhow, we have to merge these two closest cities in the distance matrix as well.
So, rows and columns are merged as the cluster is constructed.

Now how do we calculate the distance from Allahabad to the Kolkata-Dhanbad cluster? Well, there are different approaches, but
let’s assume, for example, we just select the distance from the centre of the Kolkata-Dhanbad cluster to Allahabad. Updating the distance matrix, we now haveone less cluster. Next, we look for the closest clusters once again.The next closest clusters are allahabad and delhi.So we merge them in similar ways.This process is continued till all clusters are merged and tree becomes complete.
This is a common way to implement this type of clustering, and has the benefit of caching distances between clusters.
It means, until all cities are clustered into a single cluster of size 6.

Hierarchical clustering is typically visualized as a dendrogram as shown on this slide.Each merge is represented by a horizontal line.The y-coordinate of the horizontal line is the similarity of the two clusters that were merged, where cities are viewed as singleton clusters.
By moving up from the bottom layer to the top node, a dendrogram allows us to reconstruct the history of merges that resulted in the depicted clustering.

Essentially, Hierarchical clustering does not require a pre-specified number of clusters.However, in some applications we want a partition of disjoint clusters just as in flat clustering.
 In those cases, the hierarchy needs to be cut at some point.
For example here, cutting in a specific level of similarity, we create 3 clusters of similar cities.

**ALGORITHM:**

Agglomerative clustering is a bottom-up approach.Let’s say our dataset has n data points. 
   1. we create n clusters, one for each data point. Then each point is assigned as a cluster.
   
   2. we compute the distance/proximity matrix, which will be an n * n table.
   
   3. we iteratively run the following steps until the specified cluster number is reached, or until there is only one cluster left.
   
     i) MERGE the two nearest clusters. (Distances are computed already in the proximity matrix.
     
     ii) UPDATE the proximity matrix with the new values.
     
   4. We stop after we’ve reached the specified number of clusters, or there is only one cluster remaining, with the result stored in a dendrogram. So, in the proximity matrix, we have to measure the distances between clusters, and also merge the clusters that are “nearest.”
   
So, the key operation is the computation of the proximity between the clusters with one point, and also clusters with multiple data points.

We can use different criteria to find the closest clusters, and merge them.In general, it completely depends on the data type, dimensionality of data, and most importantly,the domain knowledge of the dataset. In fact, different approaches to defining
the distance between clusters, distinguish the different algorithms.

There are multiple ways we can do this.

   1. **Single-Linkage Clustering**: Single linkage is defined as the shortest distance between 2 points in each cluster
   
   2. **Complete-Linkage Clustering**: It is the process of finding the longest distance between points in each cluster
   
   3. **Average Linkage Clustering**:This means we’re looking at the average distance of each point from one cluster to
every point in another cluster

   4. **Centroid Linkage Clustering**: Centroid is the average of the feature sets of points in a cluster. This linkage takes into account the centroid of each cluster when determining the minimum distance.
  
  **ADVANTAGES**:

   There are 3 main advantages to using hierarchical clustering:
   
1. We do not need to specify the number of clusters required for the algorithm.

2. Hierarchical clustering is easy to implement.

3. The dendrogram produced is very useful in understanding the data.

**DISADVANTAGES:**

There are some disadvantages as well. 

1. The algorithm can never undo any previous steps. So for example, the algorithm clusters 2 points,and later on we see that the connection was not a good one, the program cannot undo that step. 

2. The time complexity for the clustering can result in very long computation times, in comparison with efficient algorithms, such
k-Means. 

3. If we have a large dataset, it can become difficult to determine the correct number of clusters by the dendrogram.

**DIFFERENCES BETWEEN K-MEANS AND HIERARCHICAL CLUSTERING**:

|**K-Means**|**Hierarchical Clustering**|
|------|-------------|
|More efficient for large datasets|Does not require the number of clusters to be specified|
|k-Means gives only one partitioning of the data|Hierarchical clustering gives more than one partitioning depending on the resolution|
|k-Means returns different clusters each time it is run due to random initialization of centroids|Hierarchical clustering always generates the same clusters|
    

## DBscan (Density-Based Spatial Clustering of Applications with Noise) Clustering:
------------------------------------------------------------------------------

Most of the traditional clustering techniques, such as k-means, hierarchical, and fuzzy clustering, can be used to group data in an un-supervised way.
However, when applied to tasks with arbitrary shape clusters, or clusters within clusters,traditional techniques might not be able to achieve good results.
That is, elements in the same cluster might not share enough similarity -- or the performance may be poor.
Additionally, while partitioning-based algorithms, such as K-Means, may be easy to understand and implement in practice, the algorithm has no notion of outliers.That is, all points are assigned to a cluster, even if they do not belong in any.

In the domain of anomaly detection, this causes problems as anomalous points will be assigned to the same cluster as "normal" data points. The anomalous points pull the cluster centroid towards them, making it harder to classify them as anomalous points.

In contrast, Density-based clustering locates regions of high density that are separated from one another by regions of low density. Density, in this context, is defined as the number of points within a specified radius. A specific and very popular type of density-based clustering is DBSCAN. DBSCAN is particularly effective for tasks like class identification on a spatial context. The wonderful attribute of the DBSCAN algorithm is that it can find out any arbitrary shape cluster without getting affected by noise.

**ALGORITHM**:

It works based on 2 parameters: Radius(**R**) and Minimum Points(**M**).

**R** determines a specified radius that, if it includes enough points within it, we callit a "dense area." 

**M** determines the minimum number of data points we want in a neighborhood to define a cluster.

 1. We have to determine the type of each point. Each point in our dataset can be either :
 
**core Point**:A data point is a core point if, within R-neighborhood of the point, there are at least M points.

**border Point**:A data point is a BORDER point if:

      a. Its neighborhood contains less than M data points, or
      b. It is reachable from some core point. Here, Reachability means it is within R-distance from a core point.
      
**outlier point**:An outlier is a point that is not a core point, and also, is not close enough to be reachable from a core point.

 2. The next step is to connect core points that are neighbors, and put them in the same cluster.So, a cluster is formed as at least one core point, plus all reachable core points, plus all their borders. It simply shapes all the clusters and finds outliers as well.
 
 
 **ADVANTAGES:**
 
 1. DBSCAN can find arbitrarily shaped clusters. It can even find a cluster completely surrounded by a different cluster.
 
 2. DBSCAN has a notion of noise, and is robust to outliers.
 
 3. DBSCAN makes it very practical for use in many really world problems because it does not require one to specify the number
of clusters, such as K in k-Means.

