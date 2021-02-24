from skfeature.function.sparse_learning_based.MCFS import mcfs
from skfeature.function.sparse_learning_based.MCFS import feature_ranking as mcfs_ranking
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.function.similarity_based.lap_score import feature_ranking as lap_score_ranking
import numpy as np
from scipy.linalg import qr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sklearn.feature_selection
from traits import FeatureSelectableTrait, AutoEncoder
import torch

def choose_features(model, x_train=None, y_train=None, x_test=None, top_k=20, mode='normalized'):
    if x_train is None:
        x_train = []
    if x_test is None:
        x_test = []
    if model is not None and (hasattr(model, "model") and hasattr(model, "model_type")) and (model.model_type in ['sigmoid', 'MLP'] or issubclass(type(model.model), FeatureSelectableTrait)):
        indices, x_train, x_test = dfs_transform(model=model, x_train=x_train, x_test=x_test, k=top_k, mode=mode, verbose=True)
    elif mode == "PFA":
        indices, x_train, x_test = pfa_transform(x_train, x_test, k=top_k, debug=False, model=model)
    elif mode == "PCA":
        indices, x_train, x_test = pca(x_train, x_test, K=top_k, model=model)
    elif mode == "MCFS":
        indices, x_train, x_test = mcfs_ours(x_train, x_test, K=top_k, model=model)
    elif mode == "Lap":
        indices, x_train, x_test = lap_ours(x_train, x_test, K=top_k, model=model)
    elif mode == "lasso" or mode == "logistic_l1" or mode == "tree":
        indices, x_train, x_test = sklearn_model(model=model, x_train=x_train, x_test=x_test, k=top_k)
    elif mode in ['F', 'chi2', 'MI']:
        indices, x_train, x_test = univariate_test(x_train=x_train, x_test=x_test, 
            y_train=y_train, k=top_k, mode=mode)
    else:
        print(f"Gonna crash with {mode}")
        raise NotImplementedError

    return indices, x_train, x_test

def dfs_transform(model, x_train, x_test, k, mode, verbose=True, features_mode = 'normalized'):
    if mode == "DFS-NAS":
        features_mode = 'normalized'
    elif mode == 'DFS-NAS alphas':
        features_mode = 'alphas'
    elif mode == 'DFS-NAS weights':
        features_mode = 'weights'

    if features_mode == 'alphas':
        scores = model.alpha_feature_selectors().squeeze()
        indices = torch.topk(scores, k=k)
    elif features_mode == 'normalized':
        # NOTE IMPORTANT THOUGHT - doing abs, then mean will give different effect than doing it the other way. If a feature has different signs based on the class predicted, 
        # is it good to drop it because it is conflicting? Or keep it when it has high magnitude and thus high discriminative power?
        scores = (model.model.squash(model.alpha_feature_selectors())*torch.mean(torch.abs(model.feature_normalizers()), dim=0)).squeeze()
        indices = torch.topk(scores, k=k)
    elif features_mode == 'weights':
        scores = torch.mean(torch.abs(model.feature_normalizers()), dim=0)
        indices = torch.topk(scores, k=k)
    # x = [elem[indices.indices.cpu()] for elem in x_train]
    x = x_train[:, indices.indices.cpu().numpy()]
    # test_x = [elem[indices.indices.cpu()] for elem in x_test]
    test_x = x_test[:, indices.indices.cpu().numpy()]

    if verbose and k % 10 == 0 and features_mode == 'alphas':
        print(f"{mode} selected weights: {model.feature_normalizers().view(-1)[indices.indices]}")
        print(f"{mode} selected alphas: {model.alpha_feature_selectors().view(-1)[indices.indices]}")

    return indices, x, test_x

def univariate_test(x_train, y_train, x_test, k, mode):
    if mode == "F":
        univ = sklearn.feature_selection.f_classif 
    elif mode == "MI":
        univ = sklearn.feature_selection.mutual_info_classif
    elif mode == "chi2":
        univ = sklearn.feature_selection.chi2

    selector = sklearn.feature_selection.SelectKBest(univ, k=k).fit(x_train,y_train)
    x = selector.transform(x_train)
    x_test = selector.transform(x_test)

    return None, x, x_test

def sklearn_model(model, x_train, x_test, k):
    selector = sklearn.feature_selection.SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=k)
    x_train = selector.transform(x_train)
    test_x = selector.transform(x_test)

    return None, x_train, test_x

def pca(train, test, K, model=None):
    if model is None:
        pca = PCA(n_components = K)
        pca.fit(train)
    else:
        pca = model

    x_train = pca.transform(train)[:, :K]
    test_x = pca.transform(test)[:, :K]

    return pca, x_train, test_x

def lap_ours(train, test, K, model=None):
    if model is not None:
        indices = model[:K]
    else:
        train, test = np.array(train), np.array(test)

        scores = lap_score(train)
        indices = lap_score_ranking(scores)[: K]
    return indices, train[:, indices], test[:, indices]

def mcfs_ours(train, test, K, debug = False, model=None):
    if model is not None:
        indices = model[:K]
    else:
        train, test = np.array(train), np.array(test)
        W = mcfs(train, n_selected_features = K, verbose = debug)
        indices = mcfs_ranking(W)[: K]
        if debug:
            print(indices)
    return indices, train[:, indices], test[:, indices]

def pfa_selector(A, k, debug = False):
  class PFA(object):
      def __init__(self, n_features, q=0.5):
          self.q = q
          self.n_features = n_features

      def fit(self, X):
          if not self.q:
              self.q = X.shape[1]

          sc = StandardScaler()
          X = sc.fit_transform(X)

          pca = PCA(n_components=self.q).fit(X)
          self.n_components_ = pca.n_components_
          A_q = pca.components_.T

          kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
          clusters = kmeans.predict(A_q)
          cluster_centers = kmeans.cluster_centers_

          self.indices_ = [] 
          for cluster_idx in range(self.n_features):
            indices_in_cluster = np.where(clusters==cluster_idx)[0]
            points_in_cluster = A_q[indices_in_cluster, :]
            centroid = cluster_centers[cluster_idx]
            distances = np.linalg.norm(points_in_cluster - centroid, axis=1)
            optimal_index = indices_in_cluster[np.argmin(distances)]
            self.indices_.append(optimal_index) 
  
  pfa = PFA(n_features = k)
  pfa.fit(A)
  if debug:
    print('Performed PFW with q=', pfa.n_components_)
  column_indices = pfa.indices_
  return column_indices

def pfa_transform(train, test, k, debug = False, model=None):
    if model is not None:
        indices = model[:k]
    else:
        indices = pfa_selector(train, k, debug)
    return indices, train[:, indices], test[:, indices]