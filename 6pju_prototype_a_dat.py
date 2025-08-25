import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class GamePrototypeAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()
        self.analyzed_data = None

    def load_data(self):
        return pd.read_csv(self.data_file)

    def preprocess_data(self):
        scaler = StandardScaler()
        self.data[['numeric_col1', 'numeric_col2', ...]] = scaler.fit_transform(self.data[['numeric_col1', 'numeric_col2', ...]])
        return self.data

    def reduce_dimensions(self):
        pca = PCA(n_components=2)
        self.analyzed_data = pca.fit_transform(self.data)
        return self.analyzed_data

    def cluster_data(self):
        kmeans = KMeans(n_clusters=5)
        self.analyzed_data['cluster'] = kmeans.fit_predict(self.analyzed_data)
        return self.analyzed_data

    def visualize_results(self):
        plt.scatter(self.analyzed_data[:, 0], self.analyzed_data[:, 1], c=self.analyzed_data['cluster'])
        plt.show()

    def analyze(self):
        self.preprocess_data()
        self.reduce_dimensions()
        self.cluster_data()
        self.visualize_results()

if __name__ == "__main__":
    analyzer = GamePrototypeAnalyzer('game_data.csv')
    analyzer.analyze()