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
    
    # Ścieżka do pliku konfiguracyjnego modelu
    MODEL_CONFIG_PATH = os.getenv('MODEL_CONFIG_PATH', 'app/data/models/model_config.json')

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
    
    @classmethod
    def load_model_config(cls):
        """
        Load model configuration from the specified JSON file.
        
        Returns:
            dict: Model configuration dictionary or None if file not found or invalid
        """
        try:
            if os.path.exists(cls.MODEL_CONFIG_PATH):
                with open(cls.MODEL_CONFIG_PATH, 'r') as f:
                    return json.load(f)
            else:
                print(f"Model configuration file not found: {cls.MODEL_CONFIG_PATH}")
                return None
        except Exception as e:
            print(f"Error loading model configuration: {e}")
            return None
        
class AutoencoderModel:
    def __init__(self, input_dim, config=None):
        """
        Initialize the autoencoder model with given input dimension and optional configuration.
        
        Args:
            input_dim (int): The input dimension of the data
            config (dict, optional): Configuration dictionary for the model.
                                    If None, default configuration will be used.
        """
        # Default configuration
        self.default_config = {
            "dropout_rate": 0.2,
            "activation": "relu",
            "output_activation": "linear",
            "optimizer": "adam",
            "loss": "mean_squared_error",
            "metrics": ["accuracy"],
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu", "dropout": 0.2},
                {"type": "dense", "units": 64, "activation": "relu", "dropout": 0.2},
                {"type": "dense", "units": 32, "activation": "relu", "dropout": 0.2},
                {"type": "dense", "units": 16, "activation": "relu", "dropout": 0.0},
                {"type": "dense", "units": 32, "activation": "relu", "dropout": 0.2},
                {"type": "dense", "units": 64, "activation": "relu", "dropout": 0.2},
                {"type": "dense", "units": 128, "activation": "relu", "dropout": 0.2}
            ]
        }
        
        # Use provided config or default
        self.config = config if config is not None else self.default_config
        
        # Create autoencoder model
        self.model = self.create_autoencoder(input_dim)

    def create_autoencoder(self, input_dim):
        """
        Create an autoencoder model based on the configuration.
        
        Args:
            input_dim (int): The input dimension of the data
            
        Returns:
            tf.keras.models.Model: The created autoencoder model
        """
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        
        # Get configuration values
        layers_config = self.config.get("layers", self.default_config["layers"])
        output_activation = self.config.get("output_activation", self.default_config["output_activation"])
        optimizer = self.config.get("optimizer", self.default_config["optimizer"])
        loss = self.config.get("loss", self.default_config["loss"])
        metrics = self.config.get("metrics", self.default_config["metrics"])
        
        # Build encoder layers
        x = input_layer
        
        # The number of encoder layers is half of the total layers (rounded down)
        encoder_layers_count = len(layers_config) // 2
        
        # Build encoder
        for i in range(encoder_layers_count):
            layer_config = layers_config[i]
            units = layer_config.get("units")
            activation = layer_config.get("activation")
            dropout_rate = layer_config.get("dropout", 0.0)
            
            x = tf.keras.layers.Dense(units, activation=activation)(x)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Store the last encoder layer for encoding function
        encoded = x
        
        # Build decoder
        for i in range(encoder_layers_count, len(layers_config)):
            layer_config = layers_config[i]
            units = layer_config.get("units")
            activation = layer_config.get("activation")
            dropout_rate = layer_config.get("dropout", 0.0)
            
            x = tf.keras.layers.Dense(units, activation=activation)(x)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Output layer
        decoded = tf.keras.layers.Dense(input_dim, activation=output_activation)(x)
        
        # Create and compile the model
        autoencoder = tf.keras.models.Model(input_layer, decoded)
        autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return autoencoder
    
    def encode(self, data):
        """
        Get the encoded representation of the data.
        
        Args:
            data (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Encoded representation
        """
        # Get the number of encoder layers
        encoder_layers_count = len(self.config.get("layers", self.default_config["layers"])) // 2
        # Create encoder model that outputs the bottleneck layer
        encoder = tf.keras.models.Model(inputs=self.model.input, 
                                        outputs=self.model.layers[encoder_layers_count].output)
        return encoder.predict(data)

    def train(self, train_data, val_data, epochs, batch_size):
        """
        Train the autoencoder model.
        
        Args:
            train_data (numpy.ndarray): Training data
            val_data (numpy.ndarray): Validation data
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
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
        """
        Predict the reconstruction of the data.
        
        Args:
            data (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Reconstructed data
        """
        return self.model.predict(data)

    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        # Save the model
        self.model.save(filepath)
        
        # Save the configuration alongside the model
        config_path = filepath.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            AutoencoderModel: Loaded model instance
        """
        # Load the configuration if it exists
        config_path = filepath.replace('.h5', '_config.json')
        config = None
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration instead.")
        
        # Load the Keras model
        model = tf.keras.models.load_model(filepath, compile=False)
        
        # Create a new instance with the loaded configuration
        instance = cls(model.input_shape[1], config)
        
        # Replace the model with the loaded one
        instance.model = model
        
        # Recompile the model
        optimizer = instance.config.get("optimizer", instance.default_config["optimizer"])
        loss = instance.config.get("loss", instance.default_config["loss"])
        metrics = instance.config.get("metrics", instance.default_config["metrics"])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
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
        
        # Załaduj konfigurację modelu z pliku jeśli nie została podana w settings
        if "model_config" not in self.settings and Config.MODEL_CONFIG_PATH:
            self.settings["model_config"] = Config.load_model_config()
            if self.settings["model_config"]:
                print("Loaded model configuration from file.")
            else:
                print("Using default model configuration.")

    def calculate_anomaly_stats(self, val_data_scaled, model, threshold):
        """
        Calculate anomaly statistics based on validation data.
        
        Args:
            val_data_scaled (numpy.ndarray): Scaled validation data
            model (AutoencoderModel): Trained autoencoder model
            threshold (float): Anomaly threshold
            
        Returns:
            dict: Dictionary containing anomaly statistics
        """
        # Calculate MSE for each data point
        reconstructed = model.predict(val_data_scaled)
        mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
        
        # Determine anomalies based on threshold
        is_anomaly = mse > threshold
        
        # Calculate statistics
        total_samples = len(val_data_scaled)
        anomaly_count = np.sum(is_anomaly)
        normal_count = total_samples - anomaly_count
        
        # Calculate percentages
        anomaly_percentage = (anomaly_count / total_samples) * 100
        normal_percentage = (normal_count / total_samples) * 100
        
        # Prepare anomaly statistics
        anomaly_stats = {
            "total_samples": int(total_samples),
            "anomaly_count": int(anomaly_count),
            "normal_count": int(normal_count),
            "anomaly_percentage": float(anomaly_percentage),
            "normal_percentage": float(normal_percentage)
        }
        
        print(f"Calculated anomaly statistics: {anomaly_percentage:.2f}% anomalies, {normal_percentage:.2f}% normal samples")
        
        return anomaly_stats

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

    #Nieużywane fukcje ale warte zachowania dla przyszłych potrzeb

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

    ######

    def train(self, df, settings=None, labels_column=None):
        """
        Train the autoencoder model.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            settings (dict, optional): Custom settings for training
            labels_column (str, optional): Column name containing true anomaly label, s (1=anomaly, 0=normal)
        
        Returns:
            tuple: Trained model, training history, threshold value, and directory name
        """
        print("training")
        
        if settings:
            self.settings["columns_to_use"] = settings.get('columns_to_use', self.settings["columns_to_use"])
            self.settings["epochs"] = settings.get('epochs', self.settings["epochs"])
            self.settings["batch_size"] = settings.get('batch_size', self.settings["batch_size"])
            # Load model configuration if provided
            self.settings["model_config"] = settings.get('model_config', self.settings.get("model_config"))

        # # Sprawdź, czy mamy model bazowy do załadowania
        # if "basemodel" in self.settings:
        #     print(f"Base model specified: {self.settings['basemodel']}")
        #     return self.retrain(df, settings, labels_column)

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
        #self.save_data_versions(original_data, train_data, date_folder)
        
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        
        # Split true labels in the same way if they exist
        val_true_labels = None
        if true_labels is not None:
            _, val_true_labels = train_test_split(true_labels, test_size=0.2, random_state=42)
        
        # Save a copy of the validation data for reference when saving anomaly detection results
        val_original_data = original_data.loc[val_data.index]

        train_data_scaled = self.data_processor.scale_data(train_data)
        val_data_scaled = self.data_processor.scale_data(val_data)

        # Create model with the configuration from settings if available
        model = AutoencoderModel(train_data_scaled.shape[1], config=self.settings.get("model_config"))
        
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

        anomaly_stats = self.calculate_anomaly_stats(val_data_scaled, model, threshold)

        # Add model configuration to metadata
        training_metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': train_data.shape,
            'epochs': self.settings["epochs"],
            'batch_size': self.settings["batch_size"],
            'final_threshold': threshold,
            'columns_used': self.settings["columns_to_use"],
            'model_config': model.config,  # Save the model configuration in metadata
            'base_model_used': self.settings.get("basemodel"),  # Dodane informacje o użytym modelu bazowym
            'anomaly_statistics': anomaly_stats
        }

        metadata_path = os.path.join(date_folder, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=4, default=str)

        self.model_manager.save_model(model.model, self.data_processor.scaler, threshold, feature_importance, date_folder, model_config=model.config)

        self.save_weights_heatmap_data(model, date_folder)

        reconstructed = model.predict(val_data_scaled)
        mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
        anomaly_labels = (mse > threshold).astype(int)

        #self.save_latent_space_data(val_data_scaled, model, date_folder, val_original_data, threshold)

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
        
        # Spróbuj załadować model bazowy, jeśli określono w ustawieniach
        base_model = None
        base_model = None
        if "basefolder" in self.settings and self.settings["basefolder"]:
            basefolder_name = self.settings["basefolder"]
            try:
                # Konstruuj pełną ścieżkę do folderu
                folder_path = os.path.join("app", "data", self.settings["source"], basefolder_name)
                if os.path.exists(folder_path):
                    print(f"Loading model from folder {basefolder_name}")
                    # Załaduj model z określonego folderu
                    model_path = os.path.join(folder_path, 'autoencoder.h5')
                    scaler_path = os.path.join(folder_path, 'scaler.pkl')
                    threshold_path = os.path.join(folder_path, 'threshold.json')
                    
                    if all(os.path.exists(p) for p in [model_path, scaler_path, threshold_path]):
                        base_model = AutoencoderModel.load(model_path)
                        with open(scaler_path, 'rb') as f:
                            self.data_processor.scaler = pickle.load(f)
                        with open(threshold_path, 'r') as f:
                            threshold_data = json.load(f)
                            threshold = threshold_data.get("threshold", 0)
                        print("Loaded model from base folder successfully.")
                    else:
                        print(f"Warning: Required files not found in folder {basefolder_name}")
                else:
                    print(f"Warning: Base folder {basefolder_name} not found.")
            except Exception as e:
                print(f"Error loading base folder: {e}")
        elif "basemodel" in self.settings and self.settings["basemodel"]:
            base_model_name = self.settings["basemodel"]
            try:
                # Sprawdź, czy plik istnieje w katalogu modeli
                model_path = os.path.join("app", "data", "models", base_model_name)
                if os.path.exists(model_path) and base_model_name.endswith('.json'):
                    print(f"Loading model configuration from {base_model_name}")
                    # Załaduj konfigurację modelu z pliku JSON
                    with open(model_path, 'r') as f:
                        model_config = json.load(f)
                        self.settings["model_config"] = model_config
                        print("Loaded model configuration successfully.")
                else:
                    print(f"Warning: Base model file {base_model_name} not found or not a JSON file.")
            except Exception as e:
                print(f"Error loading base model: {e}")
        
        # Jeśli nie udało się załadować modelu bazowego, spróbuj załadować ostatni model
        if base_model is None:
            try:
                model, scaler, threshold = self.model_manager.load_latest_model()
                self.data_processor.scaler = scaler
                print("Loaded latest model from disk.")
                print(f"threshold: {threshold}")
            except Exception as e:
                print(f"Error loading existing model: {e}")
                # Jeśli nie można załadować istniejącego modelu, przekieruj do zwykłego trenowania
                return self.train(df, settings, labels_column)
            

        if base_model is None and "basefolder" not in self.settings:
            try:
                model, scaler, threshold = self.model_manager.load_latest_model()
                self.data_processor.scaler = scaler
                print("Loaded latest model from disk.")
            except Exception as e:
                print(f"Error loading existing model: {e}")
                return self.train(df, settings, labels_column)
        elif base_model is not None:
            model = base_model

        date_folder, dir_name = self.model_manager.create_date_folder()

        if settings:
            self.settings["columns_to_use"] = settings.get('columns_to_use', self.settings["columns_to_use"])
            self.settings["epochs"] = settings.get('epochs', self.settings["epochs"])
            self.settings["batch_size"] = settings.get('batch_size', self.settings["batch_size"])

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
        #self.save_data_versions(original_data, train_data, date_folder)
        
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        
        # Split true labels in the same way if they exist
        val_true_labels = None
        if true_labels is not None:
            _, val_true_labels = train_test_split(true_labels, test_size=0.2, random_state=42)
        
        # Save a copy of the validation data for reference when saving anomaly detection results
        val_original_data = original_data.loc[val_data.index]

        train_data_scaled = self.data_processor.scale_data(train_data)
        val_data_scaled = self.data_processor.scale_data(val_data)

        # Jeśli mamy konfigurację modelu, ale nie mamy załadowanego modelu, utwórz go
        if "model_config" in self.settings and base_model is None:
            model = AutoencoderModel(train_data_scaled.shape[1], config=self.settings["model_config"])
            
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

        anomaly_stats = self.calculate_anomaly_stats(val_data_scaled, model, threshold)

        # Add information to metadata
        retraining_metadata = {
            'retraining_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': train_data.shape,
            'epochs': self.settings["epochs"],
            'batch_size': self.settings["batch_size"],
            'final_threshold': threshold,
            'columns_used': self.settings["columns_to_use"],
            'base_model_used': self.settings.get("basemodel"),  # Dodane informacje o użytym modelu bazowym
            'model_config': model.config,  # Dodane informacje o konfiguracji modelu
            'anomaly_statistics': anomaly_stats
        }
        metadata_path = os.path.join(date_folder, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(retraining_metadata, f, indent=4, default=str)

        self.model_manager.save_model(model.model, self.data_processor.scaler, threshold, feature_importance, date_folder, model_config=model.config)

        self.save_weights_heatmap_data(model, date_folder)  

        reconstructed = model.predict(val_data_scaled)
        mse = np.mean(np.power(val_data_scaled - reconstructed, 2), axis=1)
        anomaly_labels = (mse > threshold).astype(int)

        #self.save_latent_space_data(val_data_scaled, model, date_folder, val_original_data, threshold)

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

    def save_model(self, model, scaler, threshold, feature_importance, folder, model_config=None):
        """
        Save model, scaler, threshold, feature importance and model configuration.
        
        Args:
            model: The trained Keras model
            scaler: The fitted scaler
            threshold: The anomaly detection threshold
            feature_importance: DataFrame with feature importance
            folder: Directory to save the model files
            model_config (dict, optional): Model configuration dictionary
        """
        model.save(os.path.join(folder, 'autoencoder.h5'))
        
        with open(os.path.join(folder, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(os.path.join(folder, 'threshold.json'), 'w') as f:
            json.dump({"threshold": threshold}, f)
        
        feature_importance.to_csv(os.path.join(folder, 'feature_importance.csv'), index=False)
        
        # Save model configuration if provided
        if model_config is not None:
            with open(os.path.join(folder, 'model_config.json'), 'w') as f:
                json.dump(model_config, f, indent=4)

    def load_latest_model(self):
        """
        Load the latest model, scaler, threshold, and model configuration.
        
        Returns:
            tuple: model, scaler, threshold, and optionally model_config
        """
        latest_folder = max(glob.glob(os.path.join(self.base_folder, '*')), key=os.path.getmtime)
        
        # Load model configuration if it exists
        model_config = None
        model_config_path = os.path.join(latest_folder, 'model_config.json')
        if os.path.exists(model_config_path):
            try:
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
            except Exception as e:
                print(f"Error loading model configuration: {e}")
        
        # Load model (with config if available)
        model = AutoencoderModel.load(os.path.join(latest_folder, 'autoencoder.h5'))
        
        with open(os.path.join(latest_folder, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(latest_folder, 'threshold.json'), 'r') as f:
            threshold = json.load(f)['threshold']
        
        return model, scaler, threshold

    def load_model_config_from_file(self, filepath):
        """
        Load model configuration from a JSON file.
        
        Args:
            filepath (str): Path to the JSON configuration file
            
        Returns:
            dict: Model configuration dictionary or None if file not found or invalid
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                print(f"Configuration file not found: {filepath}")
                return None
        except Exception as e:
            print(f"Error loading model configuration from file {filepath}: {e}")
            return None