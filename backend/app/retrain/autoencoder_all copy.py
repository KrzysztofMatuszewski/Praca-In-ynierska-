import os
import json
import pickle
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import tensorflow as tf
import glob as glob
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv

class Config:
    load_dotenv()

    COLUMNS_TO_USE = os.getenv('COLUMNS_TO_USE', '').split(',')
    COLUMNS_TO_USE_HIDS = os.getenv('COLUMNS_TO_USE_HIDS', '').split(',')

    RELATIVE_FROM = os.getenv('RELATIVE_FROM', '')
    RELATIVE_TO = os.getenv('RELATIVE_TO', '')

    TEST_SET_SIZE = int(os.getenv('TEST_SET_SIZE', 1000))
    EPOCHS = int(os.getenv('EPOCHS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))

    HIDS_CONFIG = {
        "user": os.getenv('HIDS_USER'),
        "password": os.getenv('HIDS_PASSWORD'),
        "host": os.getenv('HIDS_HOST'),
        "port": int(os.getenv('HIDS_PORT', 10200)),
        "index": os.getenv('HIDS_INDEX')
    }

    NIDS_CONFIG = {
        "user": os.getenv('NIDS_USER'),
        "password": os.getenv('NIDS_PASSWORD'),
        "host": os.getenv('NIDS_HOST'),
        "port": int(os.getenv('NIDS_PORT', 9200)),
        "index": os.getenv('NIDS_INDEX')
    }

    TRAIN_RAW_DATA = "app/data/TRAIN_RAW_DATA.csv"
    TRAIN_DATA = "app/data/TRAIN_DATA.csv"

class AutoencoderModel:
    def __init__(self, input_dim, dropout_rate=0.2):
        self.model = self.create_autoencoder(input_dim, dropout_rate)

    def create_autoencoder(self, input_dim, dropout_rate):
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
        encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
        encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
        encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
        decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
        decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
        decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
        decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = tf.keras.models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        
        return autoencoder
    
    def encode(self, data):
        encoder = tf.keras.models.Model(inputs=self.model.input, 
                                        outputs=self.model.layers[6].output)
        return encoder.predict(data)

    def train(self, train_data, val_data, epochs, batch_size):
        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        history = self.model.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, val_data),
            verbose=1,
            callbacks=[early_stop]
        )
        return history

    def predict(self, data):
        return self.model.predict(data)

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        model = tf.keras.models.load_model(filepath, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        instance = cls(model.input_shape[1])
        instance.model = model
        return instance

class AnomalyDetector:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def detect_anomalies(self, data):
        reconstructed = self.model.predict(data)
        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
        return mse, mse > self.threshold

    def calculate_threshold(self, data, percentile=95):
        reconstructed = self.model.predict(data)
        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
        return np.percentile(mse, percentile)

class FeatureImportance:
    @staticmethod
    def calculate(model, data, feature_names):
        original_mse = np.mean(np.power(data - model.predict(data), 2), axis=0)
        importance = []
        
        for i in range(data.shape[1]):
            temp_data = data.copy()
            temp_data[:, i] = 0  # Zero out the i-th feature
            perturbed_mse = np.mean(np.power(temp_data - model.predict(temp_data), 2), axis=0)
            importance.append(np.mean(perturbed_mse - original_mse))
        
        feature_imp = np.array(importance)
        return pd.DataFrame({'feature': feature_names, 'importance': feature_imp}).sort_values('importance', ascending=False)

class Autoencoder:
    def __init__(self, settings):
        self.settings = settings
        print(self.settings)
        self.data_processor = DataProcessor(self.settings["columns_to_use"])
        self.model_manager = ModelManager(self.settings["source"])

    def save_tsne_data(self, data, model, folder_path, original_data=None, perplexity=30, n_components=2):
        """
        Oblicza reprezentację t-SNE dla danych ukrytej warstwy autokodera i zapisuje unikalne wyniki w formacie JSON.
        Zoptymalizowana wersja eliminująca duplikaty punktów z pomiarem czasu wykonania.
        
        Args:
            data (numpy.ndarray): Dane wejściowe po skalowaniu
            model (AutoencoderModel): Wytrenowany model autokodera
            folder_path (str): Ścieżka do folderu, gdzie zapisać dane t-SNE
            original_data (pandas.DataFrame, optional): Oryginalne dane z nazwami kolumn
            perplexity (int): Parametr perplexity dla t-SNE (domyślnie 30)
            n_components (int): Liczba wymiarów wyjściowych (domyślnie 2)
        
        Returns:
            str: Ścieżka do zapisanego pliku JSON z danymi t-SNE
        """
        
        # Rozpocznij pomiar czasu całej funkcji
        start_time_total = time.time()
        
        # 1. Pobieranie reprezentacji ukrytej warstwy
        start_time = time.time()
        latent_representations = model.encode(data)
        encode_time = time.time() - start_time
        
        # 2. Obliczanie t-SNE dla reprezentacji ukrytej
        start_time = time.time()
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(latent_representations)
        tsne_time = time.time() - start_time
        
        # 3. Obliczanie MSE dla każdego punktu danych
        start_time = time.time()
        reconstructed = model.predict(data)
        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
        mse_time = time.time() - start_time
        
        # 4. Przygotowanie danych oryginalnych cech (pre-compute)
        start_time = time.time()
        original_features_data = []
        if original_data is not None and len(original_data) == len(data):
            for i in range(len(data)):
                features = {}
                for column in self.settings["columns_to_use"]:
                    if column in original_data.columns:
                        value = original_data.iloc[i][column]
                        features[column] = str(value)
                original_features_data.append(features)
        else:
            original_features_data = [None] * len(data)
        features_prep_time = time.time() - start_time
        
        # 5. Tworzenie hashów i deduplikacja
        start_time = time.time()
        point_hashes = {}
        for i in range(len(tsne_results)):
            # Przygotowanie danych do hasha
            coordinates = tsne_results[i].tolist()
            point_mse = float(mse[i])
            features = original_features_data[i]
            
            # Tworzenie hashowalnej reprezentacji punktu
            # Używamy sorted dla słowników, aby zapewnić spójną kolejność kluczy
            point_data = (
                tuple(coordinates),
                point_mse,
                tuple(sorted(features.items())) if features else None
            )
            
            # Tworzenie hasha
            hash_key = str(point_data)
            
            # Zapisujemy oryginalny indeks dla unikalnych punktów
            if hash_key not in point_hashes:
                point_hashes[hash_key] = i
        dedup_time = time.time() - start_time
        
        # 6. Tworzenie struktury danych JSON i zapisywanie
        start_time = time.time()
        # Tworzenie struktury danych dla pliku JSON - tylko dla unikalnych punktów
        tsne_data = {
            "points": [],
            "metadata": {
                "n_components": n_components,
                "perplexity": perplexity,
                "original_data_points": len(data),
                "unique_data_points": len(point_hashes),
                "latent_dimensions": latent_representations.shape[1],
                "features": self.settings["columns_to_use"],
                "timing": {
                    "encoding_time": encode_time,
                    "tsne_calculation_time": tsne_time,
                    "mse_calculation_time": mse_time,
                    "features_preparation_time": features_prep_time,
                    "deduplication_time": dedup_time
                }
            }
        }
        
        # Dodawanie tylko unikalnych punktów
        for _, idx in point_hashes.items():
            point = {
                "coordinates": tsne_results[idx].tolist(),
                "mse": float(mse[idx])
            }
            
            # Dodajemy oryginalne cechy jeśli są dostępne
            features = original_features_data[idx]
            if features:
                point["original_features"] = features
            
            tsne_data["points"].append(point)
        
        # Zapisywanie do pliku JSON
        tsne_path = os.path.join(folder_path, 'tsne_data.json')
        with open(tsne_path, 'w') as f:
            json.dump(tsne_data, f, indent=4)
        
        json_creation_time = time.time() - start_time
        
        # Całkowity czas wykonania
        total_time = time.time() - start_time_total
        
        # Dodaj czas zapisywania do metadanych
        tsne_data["metadata"]["timing"]["json_creation_time"] = json_creation_time
        tsne_data["metadata"]["timing"]["total_execution_time"] = total_time
        
        # Aktualizuj plik z pełnymi informacjami o czasie
        with open(tsne_path, 'w') as f:
            json.dump(tsne_data, f, indent=4)
        
        # Wyświetlenie informacji o deduplikacji i czasie wykonania
        reduction = len(data) - len(point_hashes)
        reduction_percentage = (reduction / len(data) * 100) if len(data) > 0 else 0
        print(f"Dane t-SNE zapisane w: {tsne_path}")
        print(f"Zredukowano liczbę punktów z {len(data)} do {len(point_hashes)} unikalnych punktów")
        print(f"Eliminacja duplikatów: {reduction} punktów ({reduction_percentage:.2f}%)")
        print(f"Czas wykonania całej funkcji: {total_time:.2f} sekund")
        print(f"Czasy poszczególnych etapów:")
        print(f"  - Kodowanie (encode): {encode_time:.2f} s")
        print(f"  - Obliczanie t-SNE: {tsne_time:.2f} s")
        print(f"  - Obliczanie MSE: {mse_time:.2f} s")
        print(f"  - Przygotowanie cech: {features_prep_time:.2f} s")
        print(f"  - Deduplikacja: {dedup_time:.2f} s")
        print(f"  - Tworzenie i zapis JSON: {json_creation_time:.2f} s")
        
        return tsne_path

    def save_latent_space_data(self, data, model, folder_path, original_data=None, threshold=None):
        # Pobieranie wartości z warstwy ukrytej
        latent_representations = model.encode(data)
        
        # Obliczanie MSE dla każdego punktu danych, aby oznaczyć anomalie
        reconstructed = model.predict(data)
        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
        is_anomaly = mse > threshold if threshold is not None else None
        
        # Tworzenie DataFrame z reprezentacjami ukrytymi
        latent_cols = [f"latent_dim_{i}" for i in range(latent_representations.shape[1])]
        latent_df = pd.DataFrame(data=latent_representations, columns=latent_cols)
        
        # Dodajemy MSE do danych
        latent_df['mse'] = mse
        
        # Jeśli mamy próg, dodajemy flagę anomalii
        if is_anomaly is not None:
            latent_df['is_anomaly'] = is_anomaly
        
        # Jeśli mamy oryginalne dane, dodajemy wybrane kolumny
        if original_data is not None:
            if len(original_data) == len(latent_df):
                for column in self.settings["columns_to_use"]:
                    if column in original_data.columns:
                        latent_df[column] = original_data[column].reset_index(drop=True)
        
        # Zapisujemy do pliku CSV
        latent_space_path = os.path.join(folder_path, 'latent_space_data.csv')
        latent_df.to_csv(latent_space_path, index=False)
        
        # Dodatkowo można zapisać dane do wizualizacji przestrzeni ukrytej w formacie JSON
        # do łatwiejszego użycia w aplikacjach webowych
        latent_space_json = {
            "latent_dimensions": latent_representations.shape[1],
            "data_points": len(latent_df),
            "features": self.settings["columns_to_use"],
            "anomaly_threshold": threshold
        }
        
        latent_metadata_path = os.path.join(folder_path, 'latent_space_metadata.json')
        with open(latent_metadata_path, 'w') as f:
            json.dump(latent_space_json, f, indent=4)
        
        print(f"Dane przestrzeni ukrytej zapisane w: {latent_space_path}")
        
        return latent_space_path

    def save_anomaly_detection_results(self, data, model, threshold, folder_path, original_data=None):
        """
        Save anomaly detection results with MSE for each record and the original feature values.
        
        Args:
            data (numpy.ndarray): Scaled input data
            model (AutoencoderModel): Trained autoencoder model
            threshold (float): Anomaly threshold
            folder_path (str): Path to save the anomaly detection results
            original_data (pandas.DataFrame, optional): Original unscaled data with column names
        """
        reconstructed = model.predict(data)
        mse = np.mean(np.power(data - reconstructed, 2), axis=1)
        
        # Create base DataFrame with MSE and anomaly flag
        anomaly_results_df = pd.DataFrame({
            'mse': mse,
            'is_anomaly': mse > threshold
        })
        
        # If original data is provided, add the columns_to_use values to the results
        if original_data is not None:
            # Make sure we have the same number of rows
            if len(original_data) == len(anomaly_results_df):
                # Add each column from the original data
                for column in self.settings["columns_to_use"]:
                    if column in original_data.columns:
                        anomaly_results_df[column] = original_data[column].reset_index(drop=True)
        
        anomaly_results_path = os.path.join(folder_path, 'anomaly_detection_results.csv')
        anomaly_results_df.to_csv(anomaly_results_path, index=False)
        
        # Create a summary of unique values and their anomaly counts
        self.create_unique_values_summary(anomaly_results_df, folder_path)

    def create_unique_values_summary(self, anomaly_df, folder_path):
        """
        Create a summary of unique values for each column in columns_to_use and count 
        how many are anomalies and non-anomalies.
        
        Args:
            anomaly_df (pandas.DataFrame): DataFrame with anomaly results and original columns
            folder_path (str): Path to save the summary files
        """
        summary_data = {}
        
        # For each column in columns_to_use
        for column in self.settings["columns_to_use"]:
            if column in anomaly_df.columns:
                # Initialize the summary dictionary for this column
                column_summary = {}
                
                # Group by the unique values in this column and count anomalies vs non-anomalies
                grouped = anomaly_df.groupby([column, 'is_anomaly']).size().unstack(fill_value=0)
                
                # If the 'is_anomaly' columns don't exist, create them with zeros
                if False not in grouped.columns:
                    grouped[False] = 0
                if True not in grouped.columns:
                    grouped[True] = 0
                
                # Rename columns for clarity
                grouped = grouped.rename(columns={False: 'normal', True: 'anomaly'})
                
                # Add total column
                grouped['total'] = grouped['normal'] + grouped['anomaly']
                
                # Convert to records for JSON serialization
                column_summary = grouped.reset_index().to_dict('records')
                summary_data[column] = column_summary
        
        # Save as JSON only
        json_path = os.path.join(folder_path, 'anomaly_unique_values_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=4)

    def save_loss_history(self, loss_history, folder_path):
        """
        Save loss history to a CSV file in the specified folder.
        
        Args:
            loss_history (dict): Dictionary containing loss history data
            folder_path (str): Path to save the loss history CSV
        """
        loss_history_path = os.path.join(folder_path, 'loss_history.csv')
        pd.DataFrame(loss_history).to_csv(loss_history_path, index=False)

    def save_data_versions(self, original_df, processed_df, folder_path):
        """
        Save combined original and processed (hashed) versions of the data in a single CSV
        
        Args:
            original_df: Original DataFrame before processing
            processed_df: DataFrame after processing (hashed values)
            folder_path: Path to save the data files
        """
        # Create a combined DataFrame
        combined_df = original_df.copy()
        
        # Add hashed columns with a prefix
        for column in processed_df.columns:
            combined_df[f'hashed_{column}'] = processed_df[column]
        
        # Save combined data
        combined_data_path = os.path.join(folder_path, 'input_data_combined.csv')
        combined_df.to_csv(combined_data_path, index=False)
        
        # Create a mapping file showing the relationship between original and hashed values
        mapping_data = []
        for column in original_df.columns:
            unique_mappings = dict(zip(original_df[column], processed_df[column]))
            mapping_data.append({
                'column': column,
                'mappings': unique_mappings
            })
        
        mapping_path = os.path.join(folder_path, 'hash_mappings.json')
        with open(mapping_path, 'w') as f:
            json.dump(mapping_data, f, indent=4, default=str)

    def save_weights_heatmap_data(self, model, folder_path):
        """
        Zapisuje dane wag modelu do pliku JSON w celu stworzenia mapy ciepła.
        
        Args:
            model (AutoencoderModel): Wytrenowany model autokodera
            folder_path (str): Ścieżka do folderu, w którym zostanie zapisany plik JSON
        """
        # Pobranie wag z modelu
        weights_data = {}
        
        # Przechodzenie przez wszystkie warstwy modelu
        for i, layer in enumerate(model.model.layers):
            # Pobieranie wag tylko z warstw, które mają parametry (Dense)
            if len(layer.get_weights()) > 0:
                layer_name = f"layer_{i}_{layer.name}"
                # Pobierz wagi i biasy
                weights, biases = layer.get_weights()
                
                # Przygotowanie danych do JSON dla tej warstwy
                weights_data[layer_name] = {
                    "weights": weights.tolist(),  # Konwersja na listę dla JSON
                    "biases": biases.tolist(),
                    "shape": weights.shape,
                    "input_size": weights.shape[0],
                    "output_size": weights.shape[1],
                    "layer_type": layer.__class__.__name__
                }
        
        # Dodanie informacji o cechach, aby można było je opisać na osiach mapy ciepła
        if hasattr(self, 'settings') and "columns_to_use" in self.settings:
            weights_data["features"] = self.settings["columns_to_use"]
        
        # Zapis do pliku JSON
        weights_path = os.path.join(folder_path, 'weights_heatmap_data.json')
        with open(weights_path, 'w') as f:
            json.dump(weights_data, f, indent=4)
        
        print(f"Dane wag do mapy ciepła zapisane w: {weights_path}")
        
        return weights_path

    def train(self, df, settings=None, labels_column=None):
        """
        Train the autoencoder model.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            settings (dict, optional): Custom settings for training
            labels_column (str, optional): Column name containing true anomaly labels (1=anomaly, 0=normal)
        
        Returns:
            tuple: Trained model, training history, threshold value, and directory name
        """
        print("training")
        
        if settings:
            self.settings["columns_to_use"] = settings['columns_to_use']
            self.settings["epochs"] = settings['epochs']
            self.settings["batch_size"] = settings['batch_size']

        # Extract labels if provided
        true_labels = None
        if labels_column and labels_column in df.columns:
            true_labels = df[labels_column].values
            # Remove labels column from training data if it's not in columns_to_use
            if labels_column not in self.settings["columns_to_use"]:
                df = df.drop(columns=[labels_column])

        # Get both processed and original data
        train_data, original_data = self.data_processor.preprocess(df)
        
        if train_data is None:
            raise Exception("No data")
        date_folder, dir_name = self.model_manager.create_date_folder()
        
        # Save both versions of the data
        self.save_data_versions(original_data, train_data, date_folder)
        
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        
        # Split true labels in the same way if they exist
        val_true_labels = None
        if true_labels is not None:
            _, val_true_labels = train_test_split(true_labels, test_size=0.2, random_state=42)
        
        # Save a copy of the validation data for reference when saving anomaly detection results
        val_original_data = original_data.loc[val_data.index]

        train_data_scaled = self.data_processor.scale_data(train_data)
        val_data_scaled = self.data_processor.scale_data(val_data)

        model = AutoencoderModel(train_data_scaled.shape[1])
        history = model.train(train_data_scaled, val_data_scaled, self.settings["epochs"], self.settings["batch_size"])

        loss_history = {
            'epochs': list(range(1, len(history.history['loss']) + 1)),
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        self.save_loss_history(loss_history, date_folder)

        anomaly_detector = AnomalyDetector(model, 0)  # Temporary threshold
        threshold = anomaly_detector.calculate_threshold(val_data_scaled)
        anomaly_detector.threshold = threshold

        # Pass the original validation data to include feature values in the results
        self.save_anomaly_detection_results(val_data_scaled, model, threshold, date_folder, val_original_data)

        # Generate synthetic labels if true labels don't exist
        if val_true_labels is None:
            # Generate labels based on the threshold
            reconstructed = model.predict(val_data_scaled)
            mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
            val_true_labels = (mse > threshold).astype(int)

        feature_importance = FeatureImportance.calculate(model.model, val_data_scaled, self.settings["columns_to_use"])

        # Add AUC value to metadata
        training_metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': train_data.shape,
            'epochs': self.settings["epochs"],
            'batch_size': self.settings["batch_size"],
            'final_threshold': threshold,
            'columns_used': self.settings["columns_to_use"],
        }

        metadata_path = os.path.join(date_folder, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=4)

        self.model_manager.save_model(model.model, self.data_processor.scaler, threshold, feature_importance, date_folder)

        self.save_weights_heatmap_data(model, date_folder)

        reconstructed = model.predict(val_data_scaled)
        mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
        anomaly_labels = (mse > threshold).astype(int)

        self.save_latent_space_data(val_data_scaled, model, date_folder, val_original_data, threshold)

        self.save_tsne_data(val_data_scaled, model, date_folder, val_original_data)

        return model, history, threshold, dir_name

    def retrain(self, df, settings=None, labels_column=None):
        """
        Retrain an existing autoencoder model.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            settings (dict, optional): Custom settings for training
            labels_column (str, optional): Column name containing true anomaly labels (1=anomaly, 0=normal)
        
        Returns:
            tuple: Retrained model, training history, threshold value
        """
        print("retraining")
        try:
            model, scaler, threshold = self.model_manager.load_latest_model()
            self.data_processor.scaler = scaler
        except Exception as e:
            print(f"Error loading existing model: {e}")
            return self.train(df, settings, labels_column)

        date_folder, dir_name = self.model_manager.create_date_folder()

        if settings:
            self.settings["columns_to_use"] = settings['columns_to_use']
            self.settings["epochs"] = settings['epochs']
            self.settings["batch_size"] = settings['batch_size']

        # Extract labels if provided
        true_labels = None
        if labels_column and labels_column in df.columns:
            true_labels = df[labels_column].values
            # Remove labels column from training data if it's not in columns_to_use
            if labels_column not in self.settings["columns_to_use"]:
                df = df.drop(columns=[labels_column])

        # Get both processed and original data
        train_data, original_data = self.data_processor.preprocess(df)
        
        # Save both versions of the data
        self.save_data_versions(original_data, train_data, date_folder)
        
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        
        # Split true labels in the same way if they exist
        val_true_labels = None
        if true_labels is not None:
            _, val_true_labels = train_test_split(true_labels, test_size=0.2, random_state=42)
        
        # Save a copy of the validation data for reference when saving anomaly detection results
        val_original_data = original_data.loc[val_data.index]

        train_data_scaled = self.data_processor.scale_data(train_data)
        val_data_scaled = self.data_processor.scale_data(val_data)

        history = model.train(train_data_scaled, val_data_scaled, self.settings["epochs"], self.settings["batch_size"])

        loss_history = {
            'epochs': list(range(1, len(history.history['loss']) + 1)),
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        self.save_loss_history(loss_history, date_folder)

        anomaly_detector = AnomalyDetector(model, 0)  # Temporary threshold
        threshold = anomaly_detector.calculate_threshold(val_data_scaled)
        anomaly_detector.threshold = threshold

        # Pass the original validation data to include feature values in the results
        self.save_anomaly_detection_results(val_data_scaled, model, threshold, date_folder, val_original_data)

        # Generate synthetic labels if true labels don't exist
        if val_true_labels is None:
            # Generate labels based on the threshold
            reconstructed = model.predict(val_data_scaled)
            mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
            val_true_labels = (mse > threshold).astype(int)

        feature_importance = FeatureImportance.calculate(model.model, val_data_scaled, self.settings["columns_to_use"])

        # Add AUC value to metadata
        retraining_metadata = {
            'retraining_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': train_data.shape,
            'epochs': self.settings["epochs"],
            'batch_size': self.settings["batch_size"],
            'final_threshold': threshold,
            'columns_used': self.settings["columns_to_use"],
        }
        metadata_path = os.path.join(date_folder, 'retraining_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(retraining_metadata, f, indent=4)

        self.model_manager.save_model(model.model, self.data_processor.scaler, threshold, feature_importance, date_folder)

        self.save_weights_heatmap_data(model, date_folder)  

        reconstructed = model.predict(val_data_scaled)
        mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
        anomaly_labels = (mse > threshold).astype(int)

        self.save_latent_space_data(val_data_scaled, model, date_folder, val_original_data, threshold)

        self.save_tsne_data(val_data_scaled, model, date_folder, val_original_data)

        return model, history, threshold, dir_name
    
class DataProcessor:
    def __init__(self, columns_to_use):
        self.scaler = StandardScaler()
        self.columns_to_use = columns_to_use
        print(columns_to_use)

    def preprocess(self, df):
        if len(df) == 0:
            return None, None

        df_processed = df.loc[:, self.columns_to_use]
        df_original = df_processed.copy()

        for column in self.columns_to_use:
            if column in ['source.port', 'destination.port']:
                df_processed[column] = pd.to_numeric(df_processed[column])
            df_processed[column] = df_processed[column].apply(lambda x: tuple(x) if isinstance(x, list) else x)
            df_processed[column] = pd.util.hash_pandas_object(df_processed[column], index=False, hash_key="95f9a5658c386e04")

        return df_processed, df_original

    def scale_data(self, data):
        return self.scaler.fit_transform(data)

    def inverse_scale(self, data):
        return self.scaler.inverse_transform(data)
    
class ModelManager:

    def __init__(self, source):
        self.base_folder = f"app/data/{source}"
        self.source = source

    def create_date_folder(self):
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        date_folder = os.path.join(self.base_folder, current_date)
        os.makedirs(date_folder, exist_ok=True)
        dir_name= f"{self.source}/{current_date}"
        return date_folder, dir_name

    def save_model(self, model, scaler, threshold, feature_importance, folder):
        model.save(os.path.join(folder, 'autoencoder.h5'))
        with open(os.path.join(folder, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(folder, 'threshold.json'), 'w') as f:
            json.dump({"threshold": threshold}, f)
        feature_importance.to_csv(os.path.join(folder, 'feature_importance.csv'), index=False)

    def load_latest_model(self):
        latest_folder = max(glob.glob(os.path.join(self.base_folder, '*')), key=os.path.getmtime)
        model = AutoencoderModel.load(os.path.join(latest_folder, 'autoencoder.h5'))
        with open(os.path.join(latest_folder, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(latest_folder, 'threshold.json'), 'r') as f:
            threshold = json.load(f)['threshold']
        return model, scaler, threshold
