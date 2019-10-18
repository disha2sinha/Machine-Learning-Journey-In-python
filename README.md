# Machine-Learning-Journey-In-python

**HIERARCHICAL CLUSTERING:**

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



    
