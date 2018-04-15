import numpy as np
import matplotlib.pyplot as plt

data_points = np.array([
    [0.0848835330132443, 0.206943248886680],
    [0.0738278885758684, 0.154568213233134],
    [0.238039859133728, 0.131917020763398],
    [0.454051208253475, 0.379383132540102],
    [0.276087513357917, 0.497607990564876],
    [1.43236779153440, 1.75663070089437],
    [0.0164699463749383, 0.0932857220706846],
    [0.0269314632177781, 0.390572634267382],
    [1.19376593616661, 1.55855903456055],
    [0.402531614279451, 0.0978989905133660],
    [0.225687427351724, 0.496179486589963],
    [1.91018783072814, 1.70507747511279],
    [0.191323114779979, 0.401130784882144],
    [0.394821851844845, 0.212113354951653],
    [0.182143434749897, 0.364431934025687],
    [1.49835358252355, 1.40350138880436],
    [1.80899026719904, 1.93497908617805],
    [1.35650893348105, 1.47948454563248],
    [1.5909914552748, 1.39629024850978]
])


def euclidian_dist(b, a):
    return np.linalg.norm(a-b)


def elbow(k):
    sse_hist = []
    i = 1
    while i <= k:
        sse = 0
        if i is not 0:
            clusters, centroids = k_means(i)
            for index, point in enumerate(data_points):
                pass
                cluster_point_belongs = clusters[index]
                sse += euclidian_dist(centroids[int(cluster_point_belongs)], point)
            # print("For %s means the error is %s" % (i, sse))
            sse_hist.append([i, sse])
        i += 1
    return sse_hist


def k_means(k, error=0):
    norm_dist = euclidian_dist
    num_points, num_features = data_points.shape
    prototypes = data_points[np.random.random_integers(0, num_points-1, size=k)]  # Starting points could be chosen differently
    # Contains the indexes of the prototypes
    # Can cause two warnings if the elements are not unique
    unique_num = len(np.unique(prototypes))/2
    while unique_num != k:
        prototypes = data_points[np.random.random_integers(0, num_points - 1, size=k)]
        unique_num = len(np.unique(prototypes))/2
    old_prototypes = np.zeros(prototypes.shape)
    norm = norm_dist(prototypes, old_prototypes)
    clusters = np.zeros((num_points, 1))  # this will contain values from 0 to (k-1), row i will contain
    # the cluster number that the i element from the database belongs to
    # i.e. if the first sample from our database belongs to the first cluster, clusters[0] is 0

    while norm > error:
        norm = norm_dist(prototypes, old_prototypes)
        old_prototypes = prototypes
        for point_index, point in enumerate(data_points):
            dist_vector = np.zeros((k, 1))
            for centroid_index, centroid in enumerate(prototypes):
                dist_vector[centroid_index] = norm_dist(point, centroid)
            clusters[point_index, 0] = np.argmin(dist_vector)

        temp_centroids = np.zeros((k, num_features))
        for index in range(k):
            nearest_points = [i for i in range(len(clusters)) if clusters[i] == index]
            temp_centroids[index, :] = np.mean(data_points[nearest_points], axis=0)

        prototypes = temp_centroids
    return clusters, prototypes


def plot_kmeans(clusters, centroids):  # For only when k is 2
    fig, ax = plt.subplots()
    a, b = [], []  # To get the last respective cluster points for facilitating the creation of the legend
    for i, h in enumerate(clusters):
        if int(h) == 0:
            ax.plot(data_points[i][0], data_points[i][1], 'mo')
            a = [data_points[i][0], data_points[i][1]]
        else:
            ax.plot(data_points[i][0], data_points[i][1], 'ro')
            b = [data_points[i][0], data_points[i][0]]

    ax.plot(a[0], a[0], 'mo', label='Casual user')
    ax.plot(b[0], b[1], 'ro', label='Hacker')
    ax.plot(centroids[:, 0], centroids[:, 1], 'yo', label='Centroids')
    # Specific plotting to easily plot the legend
    ax.legend()
    plt.title('K-Means, (Detect DDoS Attacks)')
    plt.xlabel('Requests to the Server')
    plt.ylabel('APM, (Actions per Minute)')
    plt.show()


def plot_elbow(sse_hist):
    n, m = np.shape(sse_hist)
    x = []
    y = []
    for i in range(m):
        for j in range(n):
            if i is 0:
                x.append(sse_hist[j][i])
            elif i is 1:
                y.append(sse_hist[j][i])

    # x = [sse_hist[:, 0]]
    # y = [sse_hist[:, 1]]
    plt.plot(x, y, '-ro')
    ratio = []
    for i in range(n - 1):
        ratio.append([(y[i] - y[i - 1]) / (y[i + 1] - y[i])])  # not very efficient
    # Could be better off selecting k by looking the graph
    ratio = np.array(ratio)
    k = np.argmax(ratio) + 1
    plt.xlabel('k, (Our Optimal k is %s)' % k)
    plt.ylabel('Sum of Squared Errors, SSE')
    plt.title('Elbow Method for Selecting k in k-means')
    plt.show()


clusters, centroids = k_means(2)
plot_kmeans(clusters, centroids)

sse_hist = elbow(15)  # If you have some spare time, put 19 instead of 15 and see its error will be 0.0
plot_elbow(sse_hist)






