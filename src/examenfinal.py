# %%
"""
# IFT870 - Examen final

Auteur : Aurélien Vauthier (19 126 456)
"""

# %%
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, make_scorer, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.spatial.distance import cdist, pdist, squareform
from itertools import combinations
from tqdm import tqdm
import numpy as np
import pandas as pd
# %matplotlib notebook
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Chargement d'un ensemble de données de faces de personnages connus
from sklearn.datasets import fetch_lfw_people

# %%
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# %%
# format des images et nombres de clusters
print("Format des images: {}".format(faces.images.shape))
print("Nombre de classes: {}".format(len(faces.target_names)))

# %%
# nombre de données par cluster
nombres = np.bincount(faces.target)
for i, (nb, nom) in enumerate(zip(nombres, faces.target_names)):
    print("{0:25} {1:3}".format(nom, nb), end='   ')
    if (i + 1) % 3 == 0:
        print()

# %%
# Affichage des 10 premières faces
fig, axes = plt.subplots(2, 5, figsize=(10, 6),
                         subplot_kw={'xticks': (), 'yticks': ()})
for nom, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(faces.target_names[nom])

# %%
# Convert data array to DataFrame and append targets
data = pd.DataFrame(faces.data)
data["target"] = faces.target

# keep the first 40 data for each target
data = data.groupby("target").head(40)

# show results
data.head()

# %%
pca = PCA(100, whiten=True, random_state=0)
data = pca.fit_transform(data.drop("target", axis=1))

# %%
def robustness(clusterings):
    len_P = 0
    same_cluster_counter = 0
    n_pairs = len(list(combinations(range(clusterings.shape[0]), 2)))

    for i, j in tqdm(combinations(range(clusterings.shape[0]), 2), total=n_pairs):
        same_cluster_count = np.sum(clusterings[i] == clusterings[j])

        same_cluster_counter += same_cluster_count
        len_P += same_cluster_count > 0

    return same_cluster_counter / (len_P * clusterings.shape[1])


# %%
"""
## Question 1 : Robustesse aux changement de paramètres d’un modèle KMeans ou AgglomerativeCLustering

*Écrivez une fonction prenant en paramètre une instance de la classe KMeans ou de la
classe AgglomerativeClustering, et retournant la robustesse de cette instance, calculée
comme suit :*

*Faire varier uniquement le paramètre n_clusters de l’instance en lui additionnant les valeurs 
`[-5,-4,-3,-2,-1,0,1,2,3,4,5]`. Pour chaque valeur du paramètre n_clusters, entraîner le modèle et
prédire un clustering. Calculer le score de robustesse R correspondant aux 11 clusterings obtenus.*
"""

# %%
def n_clusters_robustness(model):
    n_clusters_modifications = range(-5, 6)
    predictions = np.zeros((data.shape[0], len(n_clusters_modifications)))

    for i, modification in tqdm(enumerate(n_clusters_modifications), total=len(n_clusters_modifications)):
        model.n_clusters += modification
        prediction = model.fit_predict(data)
        predictions[:, i] = prediction

    return robustness(predictions)


# %%
"""
*Calculer la robustesse des modèles : KMeans(n_clusters=k, random_state=0) et
AgglomerativeClustering(n_clusters=k) pour k = 40, 60 ou 80. Quel est le modèle le plus
robuste suivant le score R ?*
"""

# %%
k_means_robustness = []
agglomerative_clustering_robustness = []

for k in range(40, 81, 20):
    kmean = KMeans(n_clusters=k, random_state=0, n_jobs=-1)
    agglo = AgglomerativeClustering(n_clusters=k)

    k_means_robustness.append(n_clusters_robustness(kmean))
    agglomerative_clustering_robustness.append(n_clusters_robustness(agglo))

sns.heatmap([k_means_robustness, agglomerative_clustering_robustness],
            xticklabels=range(40, 81, 20), yticklabels=["KMeans", "AgglomerativeClustering"], annot=True, fmt=".0%")
plt.suptitle("Scores de robustesse pour KMeans et AgglomerativeClustering")
plt.xlabel("K")
plt.ylabel("Modèle")
plt.show()

# %%
"""
## Question 2 : Robustesse aux changement de paramètres d’un modèle DBSCAN

*Écrivez une fonction prenant en paramètre une instance du modèle DBSCAN, et
retournant la robustesse de cette instance, calculée comme suit :*

*Faire varier uniquement le paramètre eps de l’instance en lui additionnant les valeurs 
`[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]`. Pour chaque valeur du paramètre eps, entraîner le
modèle et prédire un clustering. Calculer le score de robustesse R correspondant aux 11
clusterings obtenus.*
"""

# %%
def eps_robustness(model):
    eps_modifications = np.arange(-0.5, 0.6, 0.1)
    predictions = np.zeros((data.shape[0], len(eps_modifications)))

    for i, modification in tqdm(enumerate(eps_modifications), total=len(eps_modifications)):
        model.eps += modification
        prediction = model.fit_predict(data)
        predictions[:, i] = prediction

    return robustness(predictions)


# %%
"""
*Calculer la robustesse des modèles : DBSCAN(min_samples =3, eps=e) pour k = 7, 8 ou 9.
Quel est le modèle le plus robuste suivant le score R ?*
"""

# %%
dbscan_robustness = []

for eps in range(7, 10):
    dbscan = DBSCAN(min_samples=3, eps=eps, n_jobs=-1)
    dbscan_robustness.append(eps_robustness(dbscan))

sns.heatmap([dbscan_robustness], xticklabels=range(7, 10), yticklabels=["DBSCAN"], annot=True, fmt=".0%")
plt.suptitle("Scores de robustesse pour DBSCAN")
plt.xlabel("eps")
plt.ylabel("Modèle")
plt.show()

# %%
"""
## Question 3 : Robustesse à l’ajout de bruit d’un modèle KMeans ou AgglomerativeCLustering

*Écrivez une fonction prenant en paramètres une instance de la classe KMeans ou de la
classe AgglomerativeClustering et un entier X de valeur comprise entre 0 et 100
représentant un pourcentage, et retournant la robustesse de cette instance, calculée comme
suit :*

*Générer aléatoirement 10 ensembles contenant chacun $X\times1960\div100$ données (bruit) de la
même forme que les données utilisées (5655 dimensions) suivant la loi normale N(μ, σ2) pour
chaque dimension telle que μ est la moyenne de la dimension et σ2 sa variance (utiliser
numpy.random.randn par exemple). Le 11e ensemble de bruit est vide. Faire varier les
données en leur ajoutant à chaque itération un des ensembles de bruit générés. Pour chaque
itération, entraîner le modèle et prédire un clustering. Calculer le score de robustesse R
correspondant aux 11 clusterings obtenus.*
"""

# %%
def noise_generator(X):
    mu = data.mean(axis=0)
    sigma = data.var(axis=0)

    return np.random.normal(mu, sigma, (X*data.shape[0]//100, data.shape[1]))

def noise_robustness(model, X):
    predictions = np.zeros((data.shape[0], 11))

    for i in tqdm(range(10)):
        noise = noise_generator(X)
        prediction = model.fit_predict(np.concatenate((data, noise)))
        predictions[:, i] = prediction[:data.shape[0]]

    predictions[:, 10] = model.fit_predict(data)

    return robustness(predictions)


# %%
"""
*Calculer la robustesse des modèles : KMeans(n_clusters=k, random_state=0) et
AgglomerativeClustering(n_clusters=k) pour k = 40, 60 ou 80, pour une valeur X = 5. Quel
est le modèle le plus robuste suivant le score R ?*
"""

# %%
k_means_robustness = []
agglomerative_clustering_robustness = []
X = 5

for k in range(40, 81, 20):
    kmean = KMeans(n_clusters=k, random_state=0, n_jobs=-1)
    agglo = AgglomerativeClustering(n_clusters=k)

    k_means_robustness.append(noise_robustness(kmean, X))
    agglomerative_clustering_robustness.append(noise_robustness(agglo, X))

sns.heatmap([k_means_robustness, agglomerative_clustering_robustness],
            xticklabels=range(40, 81, 20), yticklabels=["KMeans", "AgglomerativeClustering"], annot=True, fmt=".0%")
plt.suptitle("Scores de robustesse pour KMeans et AgglomerativeClustering")
plt.xlabel("K")
plt.ylabel("Modèle")
plt.show()

# %%
"""
## Question 4 : Robustesse aux changement de paramètres d’un modèle DBSCAN

*Écrivez une fonction prenant en paramètre une instance du modèle DBSCAN et un
entier X de valeur comprise entre 0 et 100 représentant un pourcentage, et retournant la
robustesse de cette instance, calculée comme suit :*

*Générer aléatoirement 11 ensembles de bruit (dont 1 vide) comme indiqué à la Question 3.
Faire varier les données en leur ajoutant à chaque itération un des ensembles de bruit. Pour
chaque itération, entraîner le modèle et prédire un clustering. Calculer le score de robustesse
R correspondant aux 11 clusterings obtenus.*
"""


# %%


# %%
"""
*Calculer la robustesse des modèles : DBSCAN(min_samples =3, eps=e) pour k = 7, 8 ou 9,
pour une valeur X = 5. Quel est le modèle le plus robuste suivant le score R ?*
"""


# %%
dbscan_robustness = []

for eps in range(7, 10):
    dbscan = DBSCAN(min_samples=3, eps=eps, n_jobs=-1)
    dbscan_robustness.append(noise_robustness(dbscan, X))

sns.heatmap([dbscan_robustness], xticklabels=range(7, 10), yticklabels=["DBSCAN"], annot=True, fmt=".0%")
plt.suptitle("Scores de robustesse pour DBSCAN")
plt.xlabel("eps")
plt.ylabel("Modèle")
plt.show()

# %%
"""
## Question 5 : Modèle pour la génération du bruit

*Critiquez le modèle utilisé pour générer le bruit dans les Questions 3 et 4. Proposez un autre
modèle de bruit avec une justification du modèle.*
"""

# %%
"""

"""

# %%
