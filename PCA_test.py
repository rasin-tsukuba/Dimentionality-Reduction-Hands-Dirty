import pandas as pd
from sklearn import datasets
from Algorithms.pca import PCA

X, y = datasets.load_iris(return_X_y=True)
X = pd.DataFrame(data=X, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
y = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
print(X.head())

pca = PCA(n_components=2,
    n_iter=3,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    check_input=True,
    random_state=42)

pca = pca.fit(X)
print(pca.row_coordinates(X).head())
print(pca.explained_inertia_)
print(pca.eigenvalues_)
print(pca.total_inertia_)
print(pca.explained_inertia_)
print(pca.column_correlations(X))
print(pca.row_contributions(X).head())
print(pca.inverse_transform(pca.transform(X)).head())