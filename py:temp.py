# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% id="cLzD25UY9Xzc"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# clustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# clustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

#Outlier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

#Feature Selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE


#Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score


# outlier detection
from sklearn.ensemble import IsolationForest

# feature selection
from sklearn.feature_selection import mutual_info_classif

# classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} id="HbMNwokl8-Fn" outputId="2abfb044-796f-48d0-adb9-71b68442522c"
# upload UCI_Credit_Card.csv to files
data = pd.read_csv('UCI_Credit_Card.csv')
data.head()

# %% [markdown] id="BESDb5ka-FRd"
# # Data Preprocessing

# %% colab={"base_uri": "https://localhost:8080/"} id="Oj9Iqv85-HXe" outputId="93e8328e-518d-46d9-a025-13357266af8f"
# check if any missing information is present in a csv file.

#Number of rows before
print(f"Number of Rows before cleaning: {data.shape[0]}")

#Clean data
data = data.dropna()

#Check number of rows after:

print(f"Number of Rows after cleaning: {data.shape[0]}")

# should we also remove 0 values for certain attributes?
# Looking at the scatter plots, it looks like some values of 0 make analysis
# harder.

# %% [markdown] id="01tDVAvK-WkM"
# There are no values in the dataset that are missing in the form of NA

# %% colab={"base_uri": "https://localhost:8080/"} id="BIoEpkoL-KSa" outputId="35438e20-1563-4c89-c9a0-6c5ad694df21"
# Drop duplicates
#Number of rows before
print(f"Number of rows before removing duplicates: {data.shape[0]}")

duplicate_count = data.duplicated().sum()

print(f"Number of duplicates: {duplicate_count}")

if(duplicate_count == 0):
  print("no duplicates")

else:
  data = data.drop_duplicates()
  print(f"Number of rows after removing duplicates: {data.shape[0]}")


# %% [markdown] id="d9Z8eoiw-asq"
# We have dropped data rows that do not make sense (Example: Marriage that is listed as 1 = married, 2 = single, 3 = others, but some rows had 0. We were unsure what this meant so the rows that had it were droppped.)

# %% colab={"base_uri": "https://localhost:8080/"} id="SHdWKv6l-bs5" outputId="001ddb68-a3c0-4e3a-cfbf-3d8e9c8c5c4d"
#Drop odd values in categorical values
print(f"Number of rows before removing odd rows: {data.shape[0]}")

data = data[data['MARRIAGE'].isin([1,2,3])]
data = data[data['EDUCATION'].isin([1,2,3,4,5,6])]
data = data[data['SEX'].isin([1,2])]

pay_list = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for pay in pay_list:
  # -2 is assumed to be no credit to pay
  # -1 is credit paid fully
  # 0 is assumed to be payment made on time (minimum satement)
  data = data[data[pay].isin(range(-2,10))]


print(f"Number of rows after removing odd rows: {data.shape[0]}")

# %% id="OZfu2O5c-0ZJ"
# one-hot encoding of categorical features
cat_cols = ['MARRIAGE']
data[cat_cols] = data[cat_cols].astype('category')
data = pd.get_dummies(data, columns=cat_cols, dtype=int)


# %% colab={"base_uri": "https://localhost:8080/"} id="HyFKGXIf-1W9" outputId="214db2d2-fecf-44ab-d61e-098f9080db94"
# See new column names
print(data.columns.tolist())

# Compare before/after shape
print("Shape after encoding:", data.shape)

data.head()

data.to_csv('preprocessed_data.csv')

# %% colab={"base_uri": "https://localhost:8080/", "height": 795} id="Tm0RYnShElot" outputId="74a59ff7-3d3c-430d-a618-22d0b4533c92"
# Data augmentation generating synthetic samples
# use SMOTE to solve class imbalance problem
x = data.drop('default.payment.next.month', axis=1)
y = data['default.payment.next.month']
smote=SMOTE(sampling_strategy='minority', random_state=42)
x,y=smote.fit_resample(x,y)
plt.figure(figsize=(7,5))
sns.countplot(x=y)
plt.title('Class Distribution (0: No Default, 1: Default) after SMOTE')
plt.xlabel('Default Payment Next Month')
plt.ylabel('Count')
plt.show()
print(y.value_counts())
data = pd.concat([x, y], axis=1)
data.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="OVcV_JB9Eo0h" outputId="bcbedcc0-2fcf-480d-fe47-0ef200c813f1"
# Normalization/Standardization of data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['LIMIT_BAL','AGE'] + [f'BILL_AMT{i}' for i in range(1,7)] + [f'PAY_AMT{i}' for i in range(1,7)]
data[num_cols].describe().T[['mean', 'std']]

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="k6fTmGRGE2ok" outputId="e0f3e440-fa56-4503-e506-65edc787828c"
data[num_cols] = scaler.fit_transform(data[num_cols])
data[num_cols].describe().T[['mean', 'std']]

# %%
# save preprocessed data to csv file
data.to_csv('preprocessed_data.csv', index=False)

# %% [markdown]
# ### Read Preprocseed Data ###

# %%
data = pd.read_csv('preprocessed_data.csv')

# %% [markdown] id="pHJKwRwe9el-"
# # Clustering

# %% colab={"base_uri": "https://localhost:8080/", "height": 653} id="cuUTXKQkvs-O" outputId="dcd7090b-525f-43f7-a265-ddfe24a2bc04"
# Find optimal number of components using BIC and AIC
n_components = np.arange(1, 15)
bic_scores = []
aic_scores = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(data)
    bic_scores.append(gmm.bic(data))
    aic_scores.append(gmm.aic(data))

# Plot the BIC and AIC scores
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(n_components, bic_scores, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('BIC Score vs. Number of Components')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_components, aic_scores, marker='x')
plt.xlabel('Number of Components')
plt.ylabel('AIC Score')
plt.title('AIC Score vs. Number of Components')
plt.grid(True)
plt.legend()
plt.show()


# %%
# set up components needed for clustering
X_clustering = data.drop(['ID', 'default.payment.next.month'], axis=1)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_clustering)

tsne = TSNE(n_components=2)
tsne_components = tsne.fit_transform(X_clustering)

def visualize_cluster(components, cluster_labels, title):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=cluster_labels, palette='viridis', s=50)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Cluster')
    plt.show()


# %% [markdown] id="XGGwFLLkupOj"
# Clustering Method 1: Probabilistic Clustering - GMM

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="wNSj9Q3Jt2hA" outputId="cabe7dda-9822-4406-a856-8c4c0fa160cf"
# 1. Setup Data (Drop Target and ID for clustering)
# Ensure 'data' is your scaled/preprocessed dataframe
X_clustering = data.drop(['ID', 'default.payment.next.month'], axis=1)

# 2. Fit GMM
optimal_components = 2  # You can adjust this based on your BIC/AIC plots
gmm = GaussianMixture(n_components=optimal_components, random_state=42)
cluster_labels = gmm.fit_predict(X_clustering)

# 3. Assign Labels back to a copy of data for analysis
df_clusters = data.copy()
df_clusters['cluster'] = cluster_labels

# 4. Evaluate Performance
print("Clustering Performance Metrics:")
print(f"Silhouette Score: {silhouette_score(X_clustering, cluster_labels):.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_clustering, cluster_labels):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_clustering, cluster_labels):.4f}")

# 5. Visualize using PCA (2D)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_clustering)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=cluster_labels, palette='viridis', s=50)
plt.title('GMM Clustering Results (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# 6. Cluster Means Summary (Fixing your previous error)
# Group by the new 'cluster' column we created in df_clusters
cluster_summary = df_clusters.groupby('cluster')[X_clustering.columns].mean()
print("\nCluster Means:")
print(cluster_summary)

# %% [markdown] id="gPELu3mouyW6"
# Clustering Method 2: Hierarchical Clustering

# %% id="AsdxhG3cu5Tn"
# for this clustering algorithm, the data is too large and must be reduced
n_clusters = np.arange(2, 11)
linkages = {'ward', 'complete', 'average', 'single'}
best_silhouette = -1.0
best_labels = None
best_n = 2
best_link = None

# find best model using small sample
data_sub = pd.DataFrame(data).sample(20000, random_state=42) # Reduce to 20k samples
X_sub = data_sub.drop(['default.payment.next.month', 'ID'], axis=1)
y_sub = data_sub['default.payment.next.month']

for n in n_clusters:
    for l in linkages:
        clustering = AgglomerativeClustering(n_clusters=n, linkage=l)
        labels = clustering.fit_predict(X_sub)
        temp_silhouette_score = silhouette_score(X_sub, labels)
        if (temp_silhouette_score > best_silhouette):
            best_silhouette = temp_silhouette_score
            best_labels = labels
            best_n = n
            best_link = l

print('best number of clusters: ', best_n)
print('best linkage criterion: ', best_link)
print('silhouette score: ', best_silhouette)

chart_X_sub = PCA(2).fit_transform(X_sub)

title = f"Hierarchical Clustering with {best_n} clusters and {best_link} as linkage criterion using PCA for dimensionality reduction"
visualize_cluster(chart_X_sub, best_labels, title)

# try with t-SNE
tsne_X_sub = TSNE(2).fit_transform(X_sub)
title_tsne = f"Hierarchical Clustering with {best_n} clusters and {best_link} as linkage criterion using t-SNE for dimensionality reduction"
visualize_cluster(tsne_X_sub, best_labels, title_tsne)


# %% [markdown] id="ijOdsnlLu5pp"
# Clustering Method 3: K-Means

# %% id="ZtDS5r5su-pb"
wcss = [] 
silhouette_scores = []
k_range = range(2, 11)
X_scaled = data.drop(['ID','default.payment.next.month'], axis=1)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score
    # Silhouette score requires at least 2 clusters
    if k > 1:  
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

# Plot Silhouette Scores Method

plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different K values')
plt.grid(True)

plt.tight_layout()
plt.show()

#Use Optimal k based on the results from the earlier k findings
optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)
# Add cluster labels to original dataframe
df_with_clusters = data.copy()
df_with_clusters['Cluster'] = kmeans_clusters

print("Cluster distribution:")
print(df_with_clusters['Cluster'].value_counts().sort_index())

print("Clustering Performance Metrics:")
print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans_clusters):.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, kmeans_clusters):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, kmeans_clusters):.4f}")

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=kmeans_clusters, palette='viridis', s=50)
plt.title('K Means Clustering Results (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()


# %% [markdown] id="pyt9Hq5Au-8W"
# # Outlier Detection

# %%
def visualize_outliers(components, y_pred_outliers, title):
    plt.figure(figsize=(10, 7))
    # Plot inliers
    plt.scatter(components[y_pred_outliers == 1, 0], components[y_pred_outliers == 1, 1],
                c='blue', label='Inliers', s=50, alpha=0.6)
    # Plot outliers
    plt.scatter(components[y_pred_outliers == -1, 0], components[y_pred_outliers == -1, 1],
                c='red', label='Outliers', s=50, marker='x')

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()   


# %% [markdown] id="riUSOLmA1mZ4"
# Outlier Detection Method 1: Local Outlier Factor (LOF)

# %% colab={"base_uri": "https://localhost:8080/", "height": 676} id="6pnhHITqKBbx" outputId="58f4f0c2-657c-4b9e-a19f-c35d6ac62fb2"

# 1. Create a COPY for outlier analysis (Keeps 'data' pure)
df_outliers = data.copy()

# 2. Fit LOF
# We use X_clustering again since it only has the features we want
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_outliers = lof.fit_predict(X_clustering) # Returns -1 for outliers, 1 for inliers

# 3. Store results in the COPY
df_outliers['outlier_lof'] = y_pred_outliers

# Identify outlier/inlier sets from the copy
outliers = df_outliers[df_outliers['outlier_lof'] == -1]
inliers = df_outliers[df_outliers['outlier_lof'] == 1]

print(f"Number of outliers detected: {len(outliers)}")
print(f"Number of inliers: {len(inliers)}")

# 4. Visualize Outliers using PCA (Re-using pca_components from above)
plt.figure(figsize=(10, 7))
# Plot inliers
plt.scatter(pca_components[y_pred_outliers == 1, 0], pca_components[y_pred_outliers == 1, 1],
            c='blue', label='Inliers', s=50, alpha=0.6)
# Plot outliers
plt.scatter(pca_components[y_pred_outliers == -1, 0], pca_components[y_pred_outliers == -1, 1],
            c='red', label='Outliers', s=50, marker='x')

plt.title('Outlier Detection using LOF (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# %% [markdown] id="TmLjJ7HI18lv"
# Outlier Detection Method 2: Isolation Forset

# %% id="5o9uWRIn2GN6"
# I read https://www.geeksforgeeks.org/machine-learning/anomaly-detection-using-isolation-forest/ for help
X = data.drop(['default.payment.next.month', 'ID'], axis=1)
y = data['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
contaminations = {0.01, 0.05, 0.1, 0.125, 0.15, 0.2}

chart_X_test = PCA(2).fit_transform(X_test)
tsne_X_test = TSNE(2).fit_transform(X_test)

for contamination in contaminations:
    clf = IsolationForest(contamination=contamination)
    clf.fit(X_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # uncomment to see what each algorithm looks like
    # title = f"Isolation Forest with contamination level: {contamination} using PCA"
    # visualize_cluster(chart_X_test, y_pred_test, title)
    # title_tsne = f"Isolation Forest with contamination level: {contamination} using t-SNE"
    # visualize_cluster(tsne_X_test, y_pred_test, title_tsne)


# %%
# 0.01 looked the best
contamination = 0.01
clf = IsolationForest(contamination=contamination)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

title = f"Isolation Forest with contamination level: {contamination} using PCA"
X_test_pca = PCA(2).fit_transform(X_test)
visualize_outliers(X_test_pca, y_pred_test, title)

title_tsne = f"Isolation Forest with contamination level: {contamination} using t-SNE"
X_test_tsne = TSNE(2).fit_transform(X_test)
visualize_outliers(X_test_tsne, y_pred_test, title_tsne)

# %% [markdown] id="k76nsvqU2Gft"
# Outlier Detection Method 3: Elliptic Envolope

# %% id="7hjBovv02INT"
data_outliers = data.copy()
X_elliptic = data_outliers.drop(['ID', 'default.payment.next.month'], axis=1)
y_elliptic = data_outliers['default.payment.next.month']

outlier_results = {}

contamination_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

for contamination in contamination_levels:
    print(f"\n--- Testing contamination = {contamination} ---")
    
    # Initialize Elliptic Envelope
    ee = EllipticEnvelope(contamination=contamination, random_state=42)
    
    # Fit and predict outliers (-1 for outliers, 1 for inliers)
    outlier_labels = ee.fit_predict(X_elliptic)
    
    data_outliers['outlier_ee'] = outlier_labels
    outliers_ee = data_outliers[data_outliers['outlier_ee'] == -1]
    inliers_ee = data_outliers[data_outliers['outlier_ee'] == 1]

    print(f'Number of outliers: {len(outliers_ee)}' )
    print(f'Number of inliers: {len(inliers_ee)}')
    print(f"Outlier percentage: {(len(outliers_ee)/len(data_outliers))*100:.2f}%")

    plt.figure(figsize=(10,7))
    plt.scatter(pca_components[outlier_labels == 1, 0], pca_components[outlier_labels == 1, 1],
                c='blue', label='Inliers', s=50, alpha=0.6)
    plt.scatter(pca_components[outlier_labels == -1, 0], pca_components[outlier_labels == -1, 1],
                c='red', label='Outliers', s=50,marker='x')
    plt.title('Outlier Detection using Elliptic Envelope (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    
    data_outliers = data_outliers.drop('outlier_ee', axis=1)






# %% [markdown] id="8_lWkz6P2csa"
# # Feature Selection

# %% [markdown] id="cqfQg4Y_2eZO"
# Feature Selection Method 1: Lasso Regression

# %% colab={"base_uri": "https://localhost:8080/", "height": 755} id="55Z4OHvY2izP" outputId="4f85d6e9-7b51-413e-c2ae-f7a44094df3c"
# 1. Prepare Train/Test Data
X = data.drop(['ID', 'default.payment.next.month', 'outlier_lof', 'cluster'], axis=1, errors='ignore')
y = data['default.payment.next.month']

# 2. Apply Lasso (L1 Penalty)
# solver='liblinear' is required for L1 penalty
lasso_sel = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso_sel.fit(X, y)

# 3. Visualize Feature Importance (Coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_sel.coef_[0]
})
# Sort by absolute value of coefficient
feature_importance['Abs_Coef'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coef', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance via Lasso (L1) Logistic Regression')
plt.show()

# 4. Select Features
# Keep features where coefficient is not 0
selected_feats = feature_importance[feature_importance['Abs_Coef'] > 0]['Feature'].tolist()
print(f"Selected Features ({len(selected_feats)}): {selected_feats}")

# Update X to use only selected features for classification
X_selected = X[selected_feats]

# %% [markdown] id="f-Mpapro3D0j"
# Feature Selection Method 2: Mutual information

# %% id="CSb_0alG3Xyg"
# X is feature matrix, y is target
mi = mutual_info_classif(X, y, random_state=42)

# Convert MI scores into a readable format
mi_scores = pd.Series(mi, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': mi_scores
})

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance via Mutual Information')
plt.show()

# Keep features where mutual informatoin is greater than 0.02
selected_feats = feature_importance[feature_importance['Coefficient'] > 0.02]['Feature'].tolist()

# Update X to use only selected features for classification
X_selected_mi = X[selected_feats]

# %% [markdown] id="3FCdMYZJ3YHZ"
# Feature Selection Method 3: RFE

# %% id="EK2nddZQ3bZn"
from sklearn.feature_selection import RFE
X_RFE = data.drop(['ID', 'default.payment.next.month'], axis=1)
y_RFE = data['default.payment.next.month']

#Use logistic Regression as estimator

rfe_sel = RFE(estimator = LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=10)
rfe_sel.fit(X_RFE,y_RFE)

feature_importance = pd.DataFrame({
    'Feature' : X_RFE.columns,
    'Ranking' : rfe_sel.ranking_
})

feature_importance = feature_importance.sort_values(by='Ranking', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Ranking', y='Feature', data=feature_importance)
plt.title('Feature Rankings via RFE')
plt.show()


# %% [markdown] id="V0SnFNXx2IaY"
# # Classification

# %%
def evaluate_model(estimator, param_grid, classifier, X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                            cv=5, n_jobs=-1, scoring='f1', verbose=1)

    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_

    # 3. Evaluate Model
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Cross-Validation Consistency Check
    cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='f1')
    print(f"\n5-Fold CV F1-Scores: {cv_scores}")
    print(f"Mean CV F1-Score: {cv_scores.mean():.4f}")

    # 4. Visualizations

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{classifier} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
     


# %% [markdown] id="1_PeKLu02OKm"
# Classification Method 1: Random Forest

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="h2aDAypq2LK5" outputId="70936543-c7be-45b5-cb65-477af3a8d556"
# 1. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 2. Hyperparameter Tuning using GridSearchCV
# Defining a smaller grid to save computation time; expand if needed
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='f1', verbose=1)

print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# 3. Evaluate Model
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Cross-Validation Consistency Check
cv_scores = cross_val_score(best_rf, X_selected, y, cv=5, scoring='f1')
print(f"\n5-Fold CV F1-Scores: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f}")

# 4. Visualizations

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# %% [markdown] id="LUZg_ZEq2Rhk"
# Classification Method 2: k-NN

# %% id="39vtBLXh3o0z"
# WARNING: this cell takes a long time to run.
param_grid = {
    'n_neighbors': np.arange(2, 15),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': np.arange(25, 35)
}

knn = KNeighborsClassifier()

# rum model on regular X
evaluate_model(estimator=knn, param_grid=param_grid, classifier="k-NN", X=X)

# run model on X with only selected features (mutual information)
evaluate_model(estimator=knn, param_grid=param_grid, classifier="k-NN", X=X_selected_mi)

# %% [markdown] id="jVmZyIkO3yzQ"
# Classification Method 3:

# %% id="TdMPUdNg31IT"

# %% [markdown] id="zZhYazvLGKEQ"
# # Hyperparameter Tuning

# %% [markdown] id="pg-9mSfaGVq6"
# Grid Search for Classification Method 1: Random Forest

# %% colab={"base_uri": "https://localhost:8080/"} id="oOmE2LooGQie" outputId="fd024f8d-58e6-4da2-e1d6-478fdff3ed22"

# --- Step 1: Train Base Model (Before Tuning) ---
print("Training Base Random Forest Model (Default Parameters)...")
# Initialize with default parameters
base_rf = RandomForestClassifier(random_state=42)
base_rf.fit(X_train, y_train)

# Predict using Base Model
y_pred_base = base_rf.predict(X_test)
y_prob_base = base_rf.predict_proba(X_test)[:, 1]

# Calculate Base Metrics
base_accuracy = accuracy_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base)
base_roc = roc_auc_score(y_test, y_prob_base)

print(f"Base Model Accuracy: {base_accuracy:.4f}")
print(f"Base Model F1-Score: {base_f1:.4f}")
print(f"Base Model AUC-ROC: {base_roc:.4f}")
print("-" * 30)


# --- Step 2: Perform Hyperparameter Tuning (Grid Search) ---
print("Starting Grid Search for Hyperparameter Tuning...")

# Define the parameter grid
# We test different numbers of trees (n_estimators) and depths to find the best balance
param_grid = {
    'n_estimators': [50, 100, 200],       # Number of trees in the forest
    'max_depth': [10, 20, None],          # Max depth of the tree
    'min_samples_split': [2, 5, 10],      # Min samples required to split a node
    'min_samples_leaf': [1, 2, 4]         # Min samples required at a leaf node
}

# Initialize GridSearchCV
# cv=5 means 5-fold Cross-Validation
# n_jobs=-1 uses all available CPU cores to speed up processing
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,          # Lowered to 3 for speed, use 5 for final
                           n_jobs=-1,
                           scoring='f1',  # Optimize for F1 Score (good for imbalance)
                           verbose=2)

# Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")


# --- Step 3: Evaluate Tuned Model ---
print("-" * 30)
print("Evaluating Tuned Random Forest Model...")

# Predict using Tuned Model
y_pred_tuned = best_rf.predict(X_test)
y_prob_tuned = best_rf.predict_proba(X_test)[:, 1]

# Calculate Tuned Metrics
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned)
tuned_roc = roc_auc_score(y_test, y_prob_tuned)

print(f"Tuned Model Accuracy: {tuned_accuracy:.4f}")
print(f"Tuned Model F1-Score: {tuned_f1:.4f}")
print(f"Tuned Model AUC-ROC: {tuned_roc:.4f}")


# --- Step 4: Comparison & Discussion ---
print("\n" + "="*40)
print("IMPACT OF TUNING (Comparison)")
print("="*40)
print(f"{'Metric':<15} {'Base Model':<15} {'Tuned Model':<15} {'Improvement':<15}")
print(f"{'-'*60}")
print(f"{'Accuracy':<15} {base_accuracy:.4f}           {tuned_accuracy:.4f}           {tuned_accuracy - base_accuracy:+.4f}")
print(f"{'F1-Score':<15} {base_f1:.4f}           {tuned_f1:.4f}           {tuned_f1 - base_f1:+.4f}")
print(f"{'AUC-ROC':<15} {base_roc:.4f}           {tuned_roc:.4f}           {tuned_roc - base_roc:+.4f}")
print("="*40)


# %% [markdown] id="k-lBEUT1GdGB"
# Grid/Random Search for Classification Method 2: k-NN

# %% id="MB2nMeExGJOp"

# %% [markdown] id="HUnF02E7GgdY"
# Grid/Random Search for Classification Method 3:

# %% id="YosfK2e_GmIn"
