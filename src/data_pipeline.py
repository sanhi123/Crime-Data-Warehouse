import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 29 Indian Cities with Coordinates
CITY_COORDS = {
    'Srinagar': (34.0837, 74.7973),
    'Ludhiana': (30.9010, 75.8573),
    'Meerut': (28.9845, 77.7064),
    'Delhi': (28.7041, 77.1025),
    'Ghaziabad': (28.6692, 77.4538),
    'Faridabad': (28.4089, 77.3178),
    'Agra': (27.1767, 78.0081),
    'Jaipur': (26.9124, 75.7873),
    'Lucknow': (26.8467, 80.9462),
    'Kanpur': (26.4499, 80.3319),
    'Varanasi': (25.3176, 82.9739),
    'Patna': (25.5941, 85.1376),
    'Bhopal': (23.2599, 77.4126),
    'Indore': (22.7196, 75.8577),
    'Ahmedabad': (23.0225, 72.5714),
    'Rajkot': (22.3039, 70.8022),
    'Surat': (21.1702, 72.8311),
    'Nagpur': (21.1458, 79.0882),
    'Kolkata': (22.5726, 88.3639),
    'Nashik': (19.9975, 73.7898),
    'Thane': (19.2183, 72.9781),
    'Kalyan': (19.2403, 73.1305),
    'Mumbai': (19.0760, 72.8777),
    'Vasai': (19.3919, 72.8397),
    'Pune': (18.5204, 73.8567),
    'Visakhapatnam': (17.6868, 83.2185),
    'Hyderabad': (17.3850, 78.4867),
    'Chennai': (13.0827, 80.2707),
    'Bangalore': (12.9716, 77.5946),
}

GRID_H = 8
GRID_W = 8

class CrimeDataPipeline:
    def __init__(self, db_path='crime_warehouse.db', grid_h=GRID_H, grid_w=GRID_W):
        self.db_path = db_path
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.city_to_grid = {}
        self.grid_to_cities = {}
        self.crime_type_to_idx = {}
        self.idx_to_crime_type = {}
        self.tensor = None
        self.aggregated_tensor = None
        self.num_weeks = 0
        self.num_crime_types = 0

    def load_and_preprocess(self):
        """Loads data from sqlite warehouse, constructs 8x8 spatial grid and 4D tensor."""
        conn = sqlite3.connect(self.db_path)
        
        # Load dimension and fact tables
        df_loc = pd.read_sql('SELECT * FROM Location_Dim', conn)
        df_crime = pd.read_sql('SELECT * FROM Crime_Dim', conn)
        df_date = pd.read_sql('SELECT * FROM Date_Dim', conn)
        df_fact = pd.read_sql('''
            SELECT cf.fact_id, cf.location_dim_id, cf.crime_dim_id, cf.date_dim_id, cf.cases
            FROM Crime_Fact cf
        ''', conn)
        conn.close()

        # 1. Map Crime Types to Indices
        crime_types = sorted(df_crime['crime_type'].unique())
        self.num_crime_types = len(crime_types)
        self.crime_type_to_idx = {ct: i for i, ct in enumerate(crime_types)}
        self.idx_to_crime_type = {i: ct for i, ct in enumerate(crime_types)}
        crime_dim_map = dict(zip(df_crime['crime_dim_id'], df_crime['crime_type']))

        # 2. Build Spatial Grid (H x W)
        lats = [coords[0] for coords in CITY_COORDS.values()]
        lons = [coords[1] for coords in CITY_COORDS.values()]
        min_lat, max_lat = min(lats) - 0.5, max(lats) + 0.5
        min_lon, max_lon = min(lons) - 0.5, max(lons) + 0.5

        loc_dim_to_grid = {}
        for _, row in df_loc.iterrows():
            loc_id = row['location_dim_id']
            city = row['city']
            if city in CITY_COORDS:
                lat, lon = CITY_COORDS[city]
            else:
                lat, lon = 20.0, 78.0 # default center of India if missing
            
            # Map lat/lon to grid cells (0..grid_h-1, 0..grid_w-1)
            r = int((lat - min_lat) / (max_lat - min_lat) * (self.grid_h - 1e-5))
            c = int((lon - min_lon) / (max_lon - min_lon) * (self.grid_w - 1e-5))
            r = min(max(r, 0), self.grid_h - 1)
            c = min(max(c, 0), self.grid_w - 1)

            loc_dim_to_grid[loc_id] = (r, c)
            self.city_to_grid[city.lower()] = (r, c)
            if (r, c) not in self.grid_to_cities:
                self.grid_to_cities[(r, c)] = []
            self.grid_to_cities[(r, c)].append(city)

        # 3. Weekly Temporal Binning
        df_merged = df_fact.merge(df_date[['date_dim_id', 'year', 'month']], on='date_dim_id', how='left')
        df_merged['date_dim_id'] = df_merged['date_dim_id'].astype(int)
        
        # Determine total weeks (~260 weeks for 5 years: 2020-2024)
        # We group date_dim_id or (year, month) into sequential weekly indices
        df_merged['week_idx'] = ((df_merged['year'] - 2020) * 52 + (df_merged['date_dim_id'] % 52)).astype(int)
        
        # Normalize week_idx to start from 0
        min_week = df_merged['week_idx'].min()
        df_merged['week_idx'] = df_merged['week_idx'] - min_week
        self.num_weeks = df_merged['week_idx'].max() + 1

        # 4. Construct 4D Tensor (T, C, H, W)
        tensor_data = np.zeros((self.num_weeks, self.num_crime_types, self.grid_h, self.grid_w), dtype=np.float32)

        for _, row in df_merged.iterrows():
            w = int(row['week_idx'])
            loc_id = row['location_dim_id']
            crime_id = row['crime_dim_id']
            cases = float(row['cases']) if pd.notnull(row['cases']) else 1.0

            if loc_id in loc_dim_to_grid and crime_id in crime_dim_map:
                r, c = loc_dim_to_grid[loc_id]
                c_idx = self.crime_type_to_idx[crime_dim_map[crime_id]]
                tensor_data[w, c_idx, r, c] += cases

        self.tensor = tensor_data
        # Aggregated overall risk tensor (T, 1, H, W)
        self.aggregated_tensor = tensor_data.sum(axis=1, keepdims=True)
        
        print(f"✅ Data Pipeline Initialized:")
        print(f"   Spatial Grid: {self.grid_h}x{self.grid_w}")
        print(f"   Total Timeline: {self.num_weeks} weeks (2020-2024)")
        print(f"   Crime Categories: {self.num_crime_types}")
        print(f"   Full Tensor Shape: {self.tensor.shape} (T, C, H, W)")
        print(f"   Aggregated Tensor Shape: {self.aggregated_tensor.shape} (T, 1, H, W)")

        return self.tensor, self.aggregated_tensor

    def get_train_val_test_splits(self, seq_len=8, train_ratio=0.70, val_ratio=0.15):
        """Builds time-based sliding window samples for sequence forecasting."""
        if self.aggregated_tensor is None:
            self.load_and_preprocess()

        T = self.num_weeks
        # Create sliding sequences: X of shape (seq_len, C, H, W), Y of shape (C, H, W)
        X_all, Y_all = [], []
        for t in range(T - seq_len):
            X_all.append(self.aggregated_tensor[t : t + seq_len])
            Y_all.append(self.aggregated_tensor[t + seq_len])

        X_all = np.array(X_all) # (N_samples, seq_len, 1, H, W)
        Y_all = np.array(Y_all) # (N_samples, 1, H, W)

        N = len(X_all)
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))

        X_train, Y_train = X_all[:train_end], Y_all[:train_end]
        X_val, Y_val     = X_all[train_end:val_end], Y_all[train_end:val_end]
        X_test, Y_test   = X_all[val_end:], Y_all[val_end:]

        print(f"✅ Chronological Split:")
        print(f"   Train samples: {len(X_train)} (Weeks 0..{train_end+seq_len-1})")
        print(f"   Val samples  : {len(X_val)}")
        print(f"   Test samples : {len(X_test)}")

        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

    def train_kmeans_baseline(self, X_train, Y_train, X_test, Y_test, n_clusters=4):
        """
        Freezes a spatial K-Means baseline.
        Extracts spatial historical features per grid cell, clusters cells into risk groups,
        and logs baseline predictions for the test split.
        """
        print("\n--- Training & Freezing K-Means Baseline ---")
        # Feature vector per grid cell: mean, max, std crime count in training split
        # Y_train shape: (N_train, 1, H, W)
        cell_features = []
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                series = Y_train[:, 0, r, c]
                cell_features.append([series.mean(), series.std(), series.max(), np.median(series)])
        
        cell_features = np.array(cell_features) # (H*W, 4)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cell_features)
        
        # Compute cluster risk intensity values (mean actual target in train)
        cluster_means = {}
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                cluster_means[k] = cell_features[mask, 0].mean()
            else:
                cluster_means[k] = 0.0

        # Predict baseline for test window (constant spatial risk map per cell based on cluster assignment)
        baseline_spatial_pred = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for idx, (r, c) in enumerate([(r, c) for r in range(self.grid_h) for c in range(self.grid_w)]):
            k = labels[idx]
            baseline_spatial_pred[r, c] = cluster_means[k]

        # Expand to test sequence length: (N_test, 1, H, W)
        N_test = len(Y_test)
        Y_test_pred_kmeans = np.repeat(baseline_spatial_pred[np.newaxis, np.newaxis, :, :], N_test, axis=0)

        mae = np.abs(Y_test - Y_test_pred_kmeans).mean()
        mse = np.square(Y_test - Y_test_pred_kmeans).mean()
        print(f"✅ K-Means Baseline Frozen & Evaluated on Test Window:")
        print(f"   Test MAE: {mae:.4f}")
        print(f"   Test MSE: {mse:.4f}")

        return kmeans, Y_test_pred_kmeans


if __name__ == '__main__':
    pipeline = CrimeDataPipeline()
    tensor, agg_tensor = pipeline.load_and_preprocess()
    (X_tr, Y_tr), (X_v, Y_v), (X_te, Y_te) = pipeline.get_train_val_test_splits()
    kmeans, km_preds = pipeline.train_kmeans_baseline(X_tr, Y_tr, X_te, Y_te)
