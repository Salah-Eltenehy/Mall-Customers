#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data
data_set = pd.read_csv('Mall_Customers.csv')
cols = data_set.shape[1] #Number of Columns
X = data_set.iloc[:, 1:cols] #Independent variables

print('*'*50)
print('Data:')
print(data_set.head(10))
print('*'*50)
print('*'*50)
print('Data.describe:')
print(data_set.describe())
print('*'*50)
print('Data.info:')
print(data_set.info())
print('*'*50)

#Encoding the data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['Genre'] = label_encoder.fit_transform(X['Genre'])


#ML model 'Supervised Ml'
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
"""
#Visualisation of Elbow Method to know best number of clusters
plt.plot(range(1, 21), wcss, marker= '+', label= 'Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.title('Elbow Method')
plt.style.use('ggplot')
plt.legend()
plt.show()
"""
# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X['Annual Income (k$)'][y_kmeans == 0],
            X['Spending Score (1-100)'][y_kmeans == 0],
            c= 'r', label= 'Normal people'
            )
plt.scatter(X['Annual Income (k$)'][y_kmeans == 1],
            X['Spending Score (1-100)'][y_kmeans == 1],
            c= 'b', label= 'Standardise'
            )
plt.scatter(X['Annual Income (k$)'][y_kmeans == 2],
            X['Spending Score (1-100)'][y_kmeans == 2],
            c= 'g', label= 'Target'
            )
plt.scatter(X['Annual Income (k$)'][y_kmeans == 3],
            X['Spending Score (1-100)'][y_kmeans == 3],
            c= 'y', label= 'Carefull'
            )

plt.scatter(X['Annual Income (k$)'][y_kmeans == 4],
            X['Spending Score (1-100)'][y_kmeans == 4],
            c= '0.1', label= 'Careless'
            )
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()























