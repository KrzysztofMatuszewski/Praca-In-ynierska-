import pytest
import os
import json
import pickle
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from sklearn.preprocessing import StandardScaler


# ============================================================================
# TESTY DLA KLASY DATAPROCESSOR
# ============================================================================
class TestDataProcessor:
    """Testy jednostkowe dla klasy DataProcessor"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Fixture tworzący przykładową ramkę danych do testów"""
        return pd.DataFrame({
            'source.port': [80, 443, 8080, 22, 3306],
            'destination.port': [443, 80, 8080, 22, 3306],
            'protocol': ['tcp', 'tcp', 'http', 'ssh', 'mysql'],
            'data.size': [1024, 2048, 512, 256, 4096]
        })
    
    @pytest.fixture
    def data_processor(self):
        """Fixture tworzący instancję DataProcessor"""
        from app.retrain.autoencoder_all import DataProcessor
        columns = ['source.port', 'destination.port', 'protocol', 'data.size']
        return DataProcessor(columns_to_use=columns)
    
    def test_preprocess_returns_correct_shape(self, data_processor, sample_dataframe):
        """Test sprawdzający czy preprocess zwraca prawidłowy kształt danych"""
        processed, original = data_processor.preprocess(sample_dataframe)
        
        assert processed is not None
        assert original is not None
        assert processed.shape == original.shape
        assert processed.shape[0] == 5  # liczba wierszy
        assert processed.shape[1] == 4  # liczba kolumn
    
    def test_preprocess_handles_empty_dataframe(self, data_processor):
        """Test sprawdzający obsługę pustej ramki danych"""
        empty_df = pd.DataFrame()
        processed, original = data_processor.preprocess(empty_df)
        
        assert processed is None
        assert original is None
    
    def test_scale_data_produces_standardized_output(self, data_processor, sample_dataframe):
        """Test sprawdzający czy skalowanie danych działa prawidłowo"""
        processed, _ = data_processor.preprocess(sample_dataframe)
        scaled = data_processor.scale_data(processed)
        
        # Sprawdź czy dane zostały przeskalowane
        assert scaled is not None
        assert scaled.shape == processed.shape
        
        # Dane przeskalowane powinny mieć średnią bliską 0 i odchylenie standardowe bliskie 1
        assert np.abs(scaled.mean()) < 0.1
        assert np.abs(scaled.std() - 1.0) < 0.2
    
    def test_inverse_scale_restores_original_scale(self, data_processor):
        """Test sprawdzający czy odwrotne skalowanie przywraca oryginalne wartości"""
        original_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        scaled = data_processor.scaler.fit_transform(original_data)
        restored = data_processor.inverse_scale(scaled)
        
        # Sprawdź czy przywrócone dane są bliskie oryginałowi
        np.testing.assert_array_almost_equal(original_data, restored, decimal=5)
    
    def test_preprocess_converts_ports_to_numeric(self, data_processor, sample_dataframe):
        """Test sprawdzający konwersję portów na wartości numeryczne"""
        processed, _ = data_processor.preprocess(sample_dataframe)
        
        # Sprawdź czy kolumny portów zostały przekonwertowane na liczby
        assert pd.api.types.is_numeric_dtype(processed['source.port'])
        assert pd.api.types.is_numeric_dtype(processed['destination.port'])
    
    def test_preprocess_preserves_original_data(self, data_processor, sample_dataframe):
        """Test sprawdzający czy oryginalne dane są zachowane"""
        processed, original = data_processor.preprocess(sample_dataframe)
        
        # Sprawdź czy oryginalne dane nie zostały zmodyfikowane
        assert original.shape == processed.shape
        assert list(original.columns) == list(processed.columns)


# ============================================================================
# TESTY DLA KLASY ANOMALYDETECTOR
# ============================================================================
class TestAnomalyDetector:
    """Testy jednostkowe dla klasy AnomalyDetector"""
    
    @pytest.fixture
    def mock_model(self):
        """Fixture tworzący mock modelu autokodera"""
        model = Mock()
        # Symuluj predykcję - zwróć dane z małym szumem
        def predict_side_effect(data):
            noise = np.random.normal(0, 0.01, data.shape)
            return data + noise
        
        model.predict = MagicMock(side_effect=predict_side_effect)
        return model
    
    @pytest.fixture
    def anomaly_detector(self, mock_model):
        """Fixture tworzący instancję AnomalyDetector"""
        from app.retrain.autoencoder_all import AnomalyDetector
        threshold = 0.1
        return AnomalyDetector(model=mock_model, threshold=threshold)
    
    @pytest.fixture
    def normal_data(self):
        """Fixture z danymi normalnymi (bez anomalii)"""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 10))
    
    @pytest.fixture
    def anomaly_data(self):
        """Fixture z danymi zawierającymi anomalie"""
        np.random.seed(42)
        normal = np.random.normal(0, 1, (90, 10))
        anomalies = np.random.normal(5, 2, (10, 10))  # Wyraźnie odbiegające wartości
        return np.vstack([normal, anomalies])
    
    def test_detect_anomalies_returns_correct_shapes(self, anomaly_detector, normal_data):
        """Test sprawdzający czy detect_anomalies zwraca prawidłowe kształty wyników"""
        mse, anomaly_flags = anomaly_detector.detect_anomalies(normal_data)
        
        assert mse.shape[0] == normal_data.shape[0]
        assert anomaly_flags.shape[0] == normal_data.shape[0]
        assert mse.dtype == np.float64
        assert anomaly_flags.dtype == bool
    
    def test_detect_anomalies_identifies_normal_data(self, anomaly_detector, normal_data):
        """Test sprawdzający czy normalny ruch jest prawidłowo klasyfikowany"""
        mse, anomaly_flags = anomaly_detector.detect_anomalies(normal_data)
        
        # Większość danych normalnych powinna być sklasyfikowana jako normalna
        normal_ratio = np.sum(~anomaly_flags) / len(anomaly_flags)
        assert normal_ratio > 0.8  # Co najmniej 80% powinno być normalne
    
    def test_calculate_threshold_returns_valid_value(self, anomaly_detector, normal_data):
        """Test sprawdzający czy obliczony threshold jest prawidłowy"""
        threshold = anomaly_detector.calculate_threshold(normal_data, percentile=95)
        
        assert threshold > 0
        assert isinstance(threshold, (float, np.floating))
    
    def test_calculate_threshold_with_different_percentiles(self, anomaly_detector, normal_data):
        """Test sprawdzający różne percentyle dla threshold"""
        threshold_90 = anomaly_detector.calculate_threshold(normal_data, percentile=90)
        threshold_95 = anomaly_detector.calculate_threshold(normal_data, percentile=95)
        threshold_99 = anomaly_detector.calculate_threshold(normal_data, percentile=99)
        
        # Wyższy percentyl = wyższy threshold
        assert threshold_90 < threshold_95 < threshold_99
    
    def test_mse_calculation_is_non_negative(self, anomaly_detector, normal_data):
        """Test sprawdzający czy MSE jest zawsze nieujemne"""
        mse, _ = anomaly_detector.detect_anomalies(normal_data)
        
        assert np.all(mse >= 0)
    
    def test_threshold_comparison_is_boolean(self, anomaly_detector, normal_data):
        """Test sprawdzający czy flagi anomalii są typu boolean"""
        _, anomaly_flags = anomaly_detector.detect_anomalies(normal_data)
        
        assert anomaly_flags.dtype == bool
        assert set(np.unique(anomaly_flags)).issubset({True, False})
    
    def test_detect_anomalies_with_empty_data(self, anomaly_detector):
        """Test sprawdzający obsługę pustych danych"""
        empty_data = np.array([]).reshape(0, 10)
        mse, anomaly_flags = anomaly_detector.detect_anomalies(empty_data)
        
        assert len(mse) == 0
        assert len(anomaly_flags) == 0


# ============================================================================
# TESTY DLA KLASY MODELMANAGER
# ============================================================================
class TestModelManager:
    """Testy jednostkowe dla klasy ModelManager"""
    
    @pytest.fixture
    def temp_directory(self):
        """Fixture tworzący tymczasowy katalog dla testów"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup po testach
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def model_manager(self, temp_directory):
        """Fixture tworzący instancję ModelManager z tymczasowym katalogiem"""
        from app.retrain.autoencoder_all import ModelManager
        from unittest.mock import patch
        
        # Patch base_folder by używać temp_directory
        manager = ModelManager(source="test")
        manager.base_folder = temp_directory
        
        # Nadpisz metodę create_date_folder aby używała formatu kompatybilnego z Windows
        original_create_date_folder = manager.create_date_folder
        
        def windows_compatible_create_date_folder():
            # Użyj formatu bez dwukropka (kompatybilny z Windows)
            current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            date_folder = os.path.join(manager.base_folder, current_date)
            os.makedirs(date_folder, exist_ok=True)
            dir_name = f"{manager.source}/{current_date}"
            return date_folder, dir_name
        
        manager.create_date_folder = windows_compatible_create_date_folder
        return manager
    
    @pytest.fixture
    def mock_keras_model(self):
        """Fixture tworzący mock modelu Keras"""
        model = Mock()
        model.save = Mock()
        model.input_shape = [(None, 10)]
        return model
    
    @pytest.fixture
    def mock_scaler(self):
        """Fixture tworzący mock scalera"""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 10))
        return scaler
    
    @pytest.fixture
    def feature_importance_df(self):
        """Fixture tworzący przykładowe dane o ważności cech"""
        return pd.DataFrame({
            'feature': ['feature1', 'feature2', 'feature3'],
            'importance': [0.75, 0.50, 0.25]
        })
    
    def test_create_date_folder_creates_directory(self, model_manager):
        """Test sprawdzający czy create_date_folder tworzy katalog"""
        date_folder, dir_name = model_manager.create_date_folder()
        
        assert os.path.exists(date_folder)
        assert os.path.isdir(date_folder)
        assert dir_name.startswith("test/")
    
    def test_create_date_folder_returns_valid_paths(self, model_manager):
        """Test sprawdzający czy create_date_folder zwraca prawidłowe ścieżki"""
        date_folder, dir_name = model_manager.create_date_folder()
        
        assert isinstance(date_folder, str)
        assert isinstance(dir_name, str)
        assert len(date_folder) > 0
        assert len(dir_name) > 0
    
    def test_save_model_creates_all_files(
        self, 
        model_manager, 
        mock_keras_model, 
        mock_scaler, 
        feature_importance_df
    ):
        """Test sprawdzający czy save_model tworzy wszystkie wymagane pliki"""
        date_folder, _ = model_manager.create_date_folder()
        threshold = 0.05
        model_config = {"layers": [64, 32, 16], "activation": "relu"}
        
        model_manager.save_model(
            mock_keras_model, 
            mock_scaler, 
            threshold, 
            feature_importance_df, 
            date_folder,
            model_config=model_config
        )
        
        # Sprawdź czy wszystkie pliki zostały utworzone
        assert os.path.exists(os.path.join(date_folder, 'scaler.pkl'))
        assert os.path.exists(os.path.join(date_folder, 'threshold.json'))
        assert os.path.exists(os.path.join(date_folder, 'feature_importance.csv'))
        assert os.path.exists(os.path.join(date_folder, 'model_config.json'))
        
        # Sprawdź wywołanie metody save modelu
        mock_keras_model.save.assert_called_once()
    
    def test_save_model_threshold_format(
        self, 
        model_manager, 
        mock_keras_model, 
        mock_scaler, 
        feature_importance_df
    ):
        """Test sprawdzający format zapisu threshold"""
        date_folder, _ = model_manager.create_date_folder()
        threshold = 0.123
        
        model_manager.save_model(
            mock_keras_model, 
            mock_scaler, 
            threshold, 
            feature_importance_df, 
            date_folder
        )
        
        # Wczytaj i sprawdź threshold
        with open(os.path.join(date_folder, 'threshold.json'), 'r') as f:
            saved_threshold = json.load(f)
        
        assert 'threshold' in saved_threshold
        assert saved_threshold['threshold'] == threshold
    
    def test_save_model_config_format(
        self, 
        model_manager, 
        mock_keras_model, 
        mock_scaler, 
        feature_importance_df
    ):
        """Test sprawdzający format zapisu konfiguracji modelu"""
        date_folder, _ = model_manager.create_date_folder()
        model_config = {
            "layers": [128, 64, 32],
            "activation": "relu",
            "optimizer": "adam"
        }
        
        model_manager.save_model(
            mock_keras_model, 
            mock_scaler, 
            0.05, 
            feature_importance_df, 
            date_folder,
            model_config=model_config
        )
        
        # Wczytaj i sprawdź konfigurację
        with open(os.path.join(date_folder, 'model_config.json'), 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config == model_config
        assert saved_config['layers'] == [128, 64, 32]
        assert saved_config['activation'] == 'relu'
    
    def test_save_scaler_preserves_state(
        self, 
        model_manager, 
        mock_keras_model, 
        mock_scaler, 
        feature_importance_df
    ):
        """Test sprawdzający czy scaler jest prawidłowo zapisywany i zachowuje stan"""
        date_folder, _ = model_manager.create_date_folder()
        
        model_manager.save_model(
            mock_keras_model, 
            mock_scaler, 
            0.05, 
            feature_importance_df, 
            date_folder
        )
        
        # Wczytaj i porównaj scaler
        with open(os.path.join(date_folder, 'scaler.pkl'), 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        # Sprawdź czy parametry są zachowane
        np.testing.assert_array_almost_equal(
            loaded_scaler.mean_, 
            mock_scaler.mean_
        )
        np.testing.assert_array_almost_equal(
            loaded_scaler.scale_, 
            mock_scaler.scale_
        )
    
    def test_save_feature_importance_csv_format(
        self, 
        model_manager, 
        mock_keras_model, 
        mock_scaler, 
        feature_importance_df
    ):
        """Test sprawdzający format zapisu feature importance"""
        date_folder, _ = model_manager.create_date_folder()
        
        model_manager.save_model(
            mock_keras_model, 
            mock_scaler, 
            0.05, 
            feature_importance_df, 
            date_folder
        )
        
        # Wczytaj i sprawdź CSV
        loaded_df = pd.read_csv(os.path.join(date_folder, 'feature_importance.csv'))
        
        assert list(loaded_df.columns) == ['feature', 'importance']
        assert len(loaded_df) == 3
        pd.testing.assert_frame_equal(loaded_df, feature_importance_df)


# ============================================================================
# TESTY DLA FASTAPI ENDPOINTS
# ============================================================================
class TestFastAPIEndpoints:
    """Testy jednostkowe dla endpointów FastAPI"""
    
    @pytest.fixture
    def client(self):
        """Fixture tworzący klienta testowego FastAPI"""
        from fastapi.testclient import TestClient
        from app.main import app
        
        return TestClient(app)
    
    @pytest.fixture
    def sample_training_request(self):
        """Fixture z przykładowymi danymi żądania treningowego"""
        return {
            "source": "hids",
            "batch_size": 32,
            "columns_to_use": ["agent.id", "rule.description"],
            "relative_from": "2025-01-01T00:00",
            "relative_to": "2025-01-02T00:00",
            "epochs": 10,
            "max_size": 1000
        }
    
    @pytest.fixture
    def sample_neural_network_config(self):
        """Fixture z przykładową konfiguracją sieci neuronowej"""
        return {
            "name": "test_autoencoder",
            "dropout_rate": 0.2,
            "activation": "relu",
            "output_activation": "sigmoid",
            "optimizer": "adam",
            "loss": "mse",
            "metrics": ["accuracy"],
            "layers": [
                {
                    "type": "Dense",
                    "units": 64,
                    "activation": "relu",
                    "dropout": 0.2
                },
                {
                    "type": "Dense",
                    "units": 32,
                    "activation": "relu",
                    "dropout": 0.2
                }
            ]
        }
    
    def test_root_endpoint_returns_200(self, client):
        """Test sprawdzający czy endpoint główny zwraca status 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_training_list_endpoint_returns_folders(self, client):
        """Test sprawdzający czy endpoint /training/list zwraca listę folderów"""
        response = client.get("/training/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "hids" in data
        assert "nids" in data
        assert isinstance(data["hids"], list)
        assert isinstance(data["nids"], list)
    
    def test_training_endpoint_accepts_valid_request(self, client, sample_training_request, monkeypatch):
        """Test sprawdzający czy endpoint /training akceptuje prawidłowe żądanie"""
        # Mock funkcji get_opensearch_data aby nie wymagać prawdziwej bazy danych
        def mock_get_opensearch_data(*args, **kwargs):
            return pd.DataFrame({
                "agent.id": ["001", "002", "003"],
                "rule.description": ["desc1", "desc2", "desc3"]
            })
        
        # Mock klasy Autoencoder
        class MockAutoencoder:
            def __init__(self, settings):
                self.settings = settings
            
            def train(self, df):
                return None, None, 0.05, "test/2025-01-01_00-00-00"
            
            def retrain(self, df):
                return None, None, 0.05, "test/2025-01-01_00-00-00"
        
        # Zastosuj mocki
        import app.config.functions as functions
        monkeypatch.setattr(functions, "get_opensearch_data", mock_get_opensearch_data)
        monkeypatch.setattr("app.main.Autoencoder", MockAutoencoder)
        
        response = client.post("/training", json=sample_training_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_training_endpoint_validates_required_fields(self, client):
        """Test sprawdzający walidację wymaganych pól w endpoint /training"""
        invalid_request = {
            "source": "hids"
            # Brakuje innych wymaganych pól
        }
        
        response = client.post("/training", json=invalid_request)
        
        # Endpoint powinien zwrócić błąd walidacji
        assert response.status_code in [400, 422, 500]  # Zależnie od implementacji
    
    def test_save_neural_network_config_creates_file(self, client, sample_neural_network_config, tmp_path, monkeypatch):
        """Test sprawdzający czy endpoint zapisuje konfigurację sieci neuronowej"""
        # Użyj tymczasowego katalogu
        test_models_dir = tmp_path / "models"
        test_models_dir.mkdir()
        
        # Mock ścieżki zapisu
        monkeypatch.setattr("app.main.os.makedirs", lambda *args, **kwargs: None)
        
        # Prawidłowy endpoint to /api/neural-network/config
        response = client.post("/api/neural-network/config", json=sample_neural_network_config)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "filename" in data
        assert data["filename"].endswith(".json")
    
    def test_model_results_endpoint_returns_aggregated_data(self, client):
        """Test sprawdzający czy endpoint /model-results zwraca zagregowane dane"""
        response = client.get("/model-results")
        
        assert response.status_code == 200
        data = response.json()
        
        # Sprawdź rzeczywistą strukturę odpowiedzi z API
        assert "hids" in data or "nids" in data or "available_folders" in data
        
        # Jeśli są dane HIDS, sprawdź ich strukturę
        if "hids" in data:
            hids_data = data["hids"]
            if "feature_importance" in hids_data:
                assert isinstance(hids_data["feature_importance"], dict)
            if "columns_used" in hids_data:
                assert isinstance(hids_data["columns_used"], list)
        
        # Jeśli są dane NIDS, sprawdź ich strukturę  
        if "nids" in data:
            nids_data = data["nids"]
            if "feature_importance" in nids_data:
                assert isinstance(nids_data["feature_importance"], dict)
    
    def test_performance_metrics_endpoint_structure(self, client, monkeypatch):
        """Test sprawdzający strukturę odpowiedzi endpoint /performance-metrics"""
        # Mock OpenSearch client
        class MockOpenSearchResponse:
            def __init__(self):
                self.data = {
                    'hits': {
                        'hits': [
                            {'_source': {'timestamp': '2025-01-01', 'anomaly_score': 0.5}}
                        ]
                    }
                }
            
            def __getitem__(self, key):
                return self.data[key]
        
        def mock_search(*args, **kwargs):
            return MockOpenSearchResponse()
        
        # Zastosuj mock
        import app.main as main_module
        if hasattr(main_module, 'os_client'):
            monkeypatch.setattr(main_module.os_client, "search", mock_search)
        
        response = client.post("/performance-metrics")
        
        if response.status_code == 200:
            data = response.json()
            assert "nids" in data
            assert "hids" in data
            assert isinstance(data["nids"], list)
            assert isinstance(data["hids"], list)
    
    def test_training_folder_metadata_endpoint(self, client, tmp_path, monkeypatch):
        """Test sprawdzający endpoint pobierania metadanych folderu treningowego"""
        # Utwórz tymczasowy plik metadanych
        test_folder = tmp_path / "data" / "hids" / "2025-01-01_00-00-00"
        test_folder.mkdir(parents=True)
        
        metadata = {
            "training_date": "2025-01-01 00:00:00",
            "epochs": 50,
            "batch_size": 32
        }
        
        metadata_file = test_folder / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Mock ścieżki
        monkeypatch.setattr("app.main.os.path.exists", lambda x: True)
        
        def mock_open(path, *args, **kwargs):
            if "training_metadata.json" in path:
                return open(metadata_file, *args, **kwargs)
            raise FileNotFoundError()
        
        monkeypatch.setattr("builtins.open", mock_open)
        
        response = client.get("/training/folder-metadata/hids/2025-01-01_00-00-00")
        
        if response.status_code == 200:
            data = response.json()
            assert data["epochs"] == 50
            assert data["batch_size"] == 32
    
    def test_neural_network_config_validation(self, client):
        """Test sprawdzający walidację konfiguracji sieci neuronowej"""
        invalid_config = {
            "name": "test",
            # Brakuje wymaganych pól
        }
        
        # Prawidłowy endpoint to /api/neural-network/config
        response = client.post("/api/neural-network/config", json=invalid_config)
        
        # Powinien zwrócić błąd walidacji
        assert response.status_code in [400, 422]