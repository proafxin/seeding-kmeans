from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def process(df, processor = None, do_pca=False):
    X = df.to_numpy()
    if processor == 'scaler':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif processor == 'standard':
        standardizer = StandardScaler()
        X = standardizer.fit_transform(X)
    if do_pca == True:
        pca = PCA(
            n_components=len(df.columns),
        )
        X = pca.fit_transform(X)
    return X