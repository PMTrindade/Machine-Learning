import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy.misc import imread

RADIUS = 6371

data = pd.read_csv('century6_5.csv')
lat = data.latitude.values
lon = data.longitude.values

plt.figure(figsize=(10, 5))
plt.plot(lon, lat, '.')
plt.show()

def from_2d_to_3d(latitude, longitude):
    x = RADIUS * np.cos(latitude * np.pi/180) * np.cos(longitude * np.pi/180)
    y = RADIUS * np.cos(latitude * np.pi/180) * np.sin(longitude * np.pi/180)
    z = RADIUS * np.sin(latitude * np.pi/180)
    return x, y, z
    
X, Y, Z = from_2d_to_3d(lat, lon)

data3d = np.zeros((len(X), 3))

data3d[:,0] = X
data3d[:,1] = Y
data3d[:,2] = Z

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s = 10)
plt.show()

def plot_classes_mw(labels, filename, lon,lat):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5))    
    plt.subplot(111, projection="mollweide")
    plt.imshow(img,zorder=0,extent=[-np.pi,np.pi,-np.pi/2,np.pi/2],aspect=0.5)        
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0
    x = lon/180*np.pi
    y = lat/180*np.pi
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=0.5)
        ix = ix+1                    
    mask = np.logical_not(nots)
    if np.sum(mask) > 0:    
        plt.plot(x[mask], y[mask], 'k.', markersize=1, mew=1,markerfacecolor='w',zorder=1)
    plt.axis('off')
    plt.savefig( filename + ".png")
    plt.close
    
"""
for k=k_min to k_max
    run k-means(k)
    evaluate k.partition with validation index
"""
best_score = 0
best_NClusters = 0
best_labels = []
for i in range(2, 20):
    clusters = i
    kmeans = KMeans(n_clusters=clusters).fit(data3d)
    labels = kmeans.predict(data3d)
    score = silhouette_score(data3d, labels)
    print clusters, ": ", score
    if(score > best_score):
        best_score = score
        best_NClusters = clusters
        best_labels = labels

print "Best: ", best_NClusters, ": ", best_score
plot_classes_mw(best_labels, "KMeans" + str(best_NClusters), lon, lat)

"""""""""""""""""""""
 DENSITY BASED SCAN
"""""""""""""""""""""
distances = np.zeros(len(X))
K = 4
Param = 1
Smoothing = 10
knn = KNeighborsClassifier(n_neighbors=K).fit(data3d, distances)
distances, _ = knn.kneighbors()
distances = distances.max(1)

# SORTING
distances = np.sort(distances)

# SCALING
max_dist = np.amax(distances)
distances = distances/float(max_dist)

"""
dY = Yi - Yi-1
dX = Xi - Xi-1 = 1/N (N - data points)

first where derivative is higher than a certain value - 1
"""
dY = np.diff(distances[::Smoothing])
dX = 1/float(len(distances[::Smoothing]))

derivative = dY/dX

def get_elbow_index(derivative):
    for i in range(len(derivative)):
        if(derivative[i] > Param):
            return i
            
elbow = distances[get_elbow_index(derivative) * Smoothing]


plt.figure(figsize=(10,10))
plt.plot(range(len(distances)), distances, '-')
plt.axhline(y=elbow)
plt.show()

eps = elbow * max_dist

labels = DBSCAN(eps=eps, min_samples=K).fit_predict(data3d)
score = silhouette_score(data3d, labels)

print "DBSCAN: ", K, ": ", score

plot_classes_mw(labels, "DBScan" + str(K), lon, lat)
