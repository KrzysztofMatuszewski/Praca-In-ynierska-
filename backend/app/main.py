import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import json 
import os
import pandas as pd
from datetime import datetime
import sys
from app.config import  functions
from app.config import Config
import requests
from threading import Lock
from sklearn.metrics import roc_curve, auc
import numpy as np
import redis
import csv
from opensearchpy import OpenSearch, helpers
from fastapi.middleware.cors import CORSMiddleware
import logging
from tensorflow import keras
import pickle
from pydantic import BaseModel, Field
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrain.autoencoder_all import (
    Autoencoder
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/scripts/graph_strona/static"), name="static")
templates = Jinja2Templates(directory="app/scripts/graph_strona/templates")

nids_encoder_base_path = os.path.relpath("app/data/nids/")
hids_encoder_base_path = os.path.relpath("app/data/hids/")

encoder_lock = Lock()

# Initialize the AutoencoderSystem
config = Config()

class EncoderInitializer:
    def __init__(self, event_source):
        self.event_source = event_source
        self.scaler = None
        self.threshold = None
        self.autoencoder = None
        self.initialize()

    def initialize(self):
        try:
            # Determine the base path based on event source
            encoder_base_path = (
                hids_encoder_base_path if self.event_source == "hids" 
                else nids_encoder_base_path
            )

            # Find the latest folder
            latest_folder = get_latest_folder(encoder_base_path)
            print(f"Latest folder found: {latest_folder}")

            # Load scaler
            scaler_path = os.path.join(latest_folder, 'scaler.pkl')
            print(f"Loading scaler from: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load autoencoder
            autoencoder_path = os.path.join(latest_folder, 'autoencoder.h5')
            print(f"Loading autoencoder from: {autoencoder_path}")
            self.autoencoder = keras.models.load_model(autoencoder_path)

            # Load threshold
            threshold_path = os.path.join(latest_folder, 'threshold.json')
            print(f"Loading threshold from: {threshold_path}")
            with open(threshold_path, "r") as f:
                threshold_data = json.load(f)
                self.threshold = threshold_data["threshold"]

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            self.scaler = None
            self.threshold = None
            self.autoencoder = None

    def get_components(self):
        if self.scaler is None or self.threshold is None or self.autoencoder is None:
            return None
        
        return {
            "scaler": self.scaler,
            "threshold": self.threshold,
            "autoencoder": self.autoencoder
        }
    
class Agent(BaseModel):
    id: Union[str, List[str]]

class NormalFilterHids(BaseModel):
    agent: Agent

class PortRange(BaseModel):
    from_: int
    to: int

class EndpointConfig(BaseModel):
    ip: str
    port: PortRange

class NormalFilterNids(BaseModel):
    source: EndpointConfig
    destination: EndpointConfig

class TrainRequestHids(BaseModel):
    normal_filters: List[NormalFilterHids]

class TrainRequestNids(BaseModel):
    normal_filters: List[NormalFilterNids]

class Layer(BaseModel):
    type: str
    units: int
    activation: str
    dropout: float

class NeuralNetworkConfig(BaseModel):
    name: str = Field(..., description="Nazwa sieci neuronowej") 
    dropout_rate: float
    activation: str
    output_activation: str
    optimizer: str
    loss: str
    metrics: List[str]
    layers: List[Layer]

os_auth = (config.NIDS_CONFIG["user"], config.NIDS_CONFIG["password"])

os_client = OpenSearch(
    hosts=[{'host': config.NIDS_CONFIG["host"], 'port': config.NIDS_CONFIG["port"]}],
    http_compress=True,  # enables gzip compression for request bodies
    http_auth=os_auth,
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=1000
)

def read_csv(file_path):
    with open(file_path, 'r') as file:
        return list(csv.reader(file))

def read_feature_importance(file_path):
    data = read_csv(file_path)[1:]  # Skip header
    features, importances = zip(*[(row[0], float(row[1])) for row in data])
    return list(features), list(importances)

@app.get("/training/folder-metadata/{source}/{folder_name}")
async def get_folder_metadata(source: str, folder_name: str):
    try:
        metadata_path = f"app/data/{source}/{folder_name}/training_metadata.json"
        
        if not os.path.exists(metadata_path):
            return JSONResponse(
                content={"error": "Metadata file not found"},
                status_code=404
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return JSONResponse(
            content=metadata,
            status_code=200
        )
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/training/models/list")
async def list_models():
    try:
        models_path = "app/data/models"
        hids_path = "app/data/hids"
        nids_path = "app/data/nids"
        
        # Create directories if they don't exist
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(hids_path, exist_ok=True)
        os.makedirs(nids_path, exist_ok=True)
        
        # Get all files in the models directory
        files = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
        
        # Get folder names in the hids and nids directories
        hids_folders = [f for f in os.listdir(hids_path) if os.path.isdir(os.path.join(hids_path, f))]
        nids_folders = [f for f in os.listdir(nids_path) if os.path.isdir(os.path.join(nids_path, f))]
        
        # For each file, get creation date and additional metadata
        models_data = []
        for file in files:
            file_path = os.path.join(models_path, file)
            creation_time = os.path.getctime(file_path)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Read configuration data if it's a JSON file
            config_data = None
            if file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        config_data = json.load(f)
                except Exception as e:
                    print(f"Error reading config file {file}: {str(e)}")
                    config_data = {"error": str(e)}
            
            models_data.append({
                "filename": file,
                "created_at": creation_date,
                "config": config_data
            })
        
        # Sort models by creation date (newest first)
        models_data.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Get and sort hids folders by creation date
        hids_folders_data = []
        for folder in hids_folders:
            folder_path = os.path.join(hids_path, folder)
            creation_time = os.path.getctime(folder_path)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            hids_folders_data.append({
                "folder_name": folder,
                "created_at": creation_date
            })
        # Sort hids folders by creation date (newest first)
        hids_folders_data.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Get and sort nids folders by creation date
        nids_folders_data = []
        for folder in nids_folders:
            folder_path = os.path.join(nids_path, folder)
            creation_time = os.path.getctime(folder_path)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            nids_folders_data.append({
                "folder_name": folder,
                "created_at": creation_date
            })
        # Sort nids folders by creation date (newest first)
        nids_folders_data.sort(key=lambda x: x["created_at"], reverse=True)

        return JSONResponse(
            content={
                "models": models_data,
                "hids_folders": hids_folders_data,
                "nids_folders": nids_folders_data
            },
            status_code=200
        )
    except Exception as e:
        print(f"Error listing models and folders: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/api/neural-network/config")
async def save_neural_network_config(config: NeuralNetworkConfig):
    """
    Endpoint do odbierania konfiguracji sieci neuronowej z frontendu.
    """
    try:
        # Tworzymy folder dla konfiguracji jeśli nie istnieje
        config_dir = os.path.join("app", "data", "models")
        os.makedirs(config_dir, exist_ok=True)
        
        # Używamy podanej nazwy lub generujemy domyślną, jeśli nie podano
        if config.name:
            # Usuwamy niedozwolone znaki z nazwy pliku
            safe_name = re.sub(r'[^\w\-_.]', '_', config.name)
            filename = f"{safe_name}.json"
        else:
            # Generujemy unikalną nazwę pliku na podstawie aktualnej daty i czasu
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"nn_config_{timestamp}.json"
            
        file_path = os.path.join(config_dir, filename)
        
        # Zapisujemy konfigurację do pliku JSON
        with open(file_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        
        # Tutaj możesz dodać kod do bezpośredniego wykorzystania konfiguracji
        # np. inicjalizacja modelu, przekazanie do modułu treningowego itp.
        
        print(f"Zapisano konfigurację sieci neuronowej: {file_path}")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Konfiguracja sieci neuronowej została zapisana",
                "filename": filename
            },
            status_code=200
        )
    except Exception as e:
        print(f"Błąd podczas zapisywania konfiguracji sieci neuronowej: {str(e)}")
        return JSONResponse(
            content={'error': str(e)},
            status_code=500
        )

@app.post("/performance-metrics")
async def get_performance_metrics():
    try:
        # Get data from the last hour
        nids_index = f"{config.NIDS_CONFIG['index']}-*"
        hids_index = f"{config.HIDS_CONFIG['index']}-*"
        
        query = {
            "size": 100,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": "now-1h"
                                }
                            }
                        },
                        {
                            "exists": {
                                "field": "anomaly_detection"
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    "@timestamp": {
                        "order": "desc"
                    }
                }
            ]
        }

        # Get both NIDS and HIDS events
        nids_response = os_client.search(index=nids_index, body=query)
        hids_response = os_client.search(index=hids_index, body=query)
        
        return JSONResponse(
            content={
                'nids': [hit['_source'] for hit in nids_response['hits']['hits']],
                'hids': [hit['_source'] for hit in hids_response['hits']['hits']]
            },
            status_code=200
        )

    except Exception as e:
        print(f"Error fetching performance metrics: {e}")
        return JSONResponse(
            content={'error': str(e)},
            status_code=500
        )
    
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/training/data")
async def training_data(request: Request):
    try:
        data = await request.json()
        #print(data)
        folder = "app/data/"+data["source"]+"/"+data["folder_name"]
        #print(folder)

        if not os.path.exists(folder):
            raise ValueError(f"The base path does not exist: {folder}")
        
        feature_importance = read_feature_importance(f"{folder}/feature_importance.csv")
        #print(feature_importance)
        response = {
            "feature": feature_importance[0],
            "importance": feature_importance[1],
            
        }
        return JSONResponse(content = response,status_code = 200)
    except Exception as e:
        print(f"Error fetching performance metrics: {e}")
        return JSONResponse(
            content={'error': str(e)},
            status_code=500
        )

@app.get("/training/list")
async def training_list(request: Request):
    try:
        base_path_hids = "app/data/hids"
        base_path_nids = "app/data/nids"
        folders_hids = [f for f in os.listdir(base_path_hids) if os.path.isdir(os.path.join(base_path_hids, f))]
        folders_nids = [f for f in os.listdir(base_path_nids) if os.path.isdir(os.path.join(base_path_nids, f))]
        response = {
            "hids": folders_hids,
            "nids": folders_nids
        }

        return JSONResponse(content = response,status_code = 200)
    except Exception as e:
        print(f"Error fetching performance metrics: {e}")
        return JSONResponse(
            content={'error': str(e)},
            status_code=500
        )

@app.post("/training")
async def training(request: Request):
    try:
        global df
        data = await request.json()
        relative_from = data["relative_from"]
        relative_to = data["relative_to"]
        print(relative_from)
        print(relative_to)

        query = {
                "size": data["max_size"],
                "sort":[
                    {
                        "@timestamp":{
                            "order": "asc"
                        }
                    }
                ],
                "query": {
                    "bool": {
                        "filter":{
                            "bool":{
                                "must": [
                                    {
                                        "range": {
                                            "@timestamp": {
                                                "gte": relative_from,
                                                "lt": relative_to
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        print(query)

        if data["source"] == "nids":
            print(data["source"])
            df = functions.get_opensearch_data(config.NIDS_CONFIG, query, data["columns_to_use"])
            #print(df)
        if data["source"] == "hids":
            print(data["source"])
            df = functions.get_opensearch_data(config.HIDS_CONFIG, query, data["columns_to_use"])
            #print(df)
        
        settings = {
            "columns_to_use": data["columns_to_use"],
            "epochs": data["epochs"],
            "batch_size": data["batch_size"],
            "source": data["source"]
        }

        # Sprawdź, czy przekazano nazwę modelu bazowego
        basemodel = data.get("basemodel")
        basefolder = data.get("basefolder")


        if basemodel:
            print(f"Using base model: {basemodel}")
            settings["basemodel"] = basemodel
            
            # Wczytaj konfigurację modelu bazowego jeśli istnieje
            model_config_path = os.path.join("app", "data", "models", basemodel)
            if os.path.exists(model_config_path) and basemodel.endswith('.json'):
                try:
                    with open(model_config_path, 'r') as f:
                        model_config = json.load(f)
                        settings["model_config"] = model_config
                        print(f"Loaded model configuration from {basemodel}")
                except Exception as e:
                    print(f"Error loading model configuration: {str(e)}")
            elif basefolder:
                print(f"Using base folder for retrain: {basefolder}")
                settings["basefolder"] = basefolder


        #print(df)
        autoencoder = Autoencoder(settings)
        
        # Jeśli mamy model bazowy, użyjmy go do retrenowania
        # if basemodel:
        #     model, history, threshold, dir_name = autoencoder.retrain(df)
        # else:
        #     model, history, threshold, dir_name = autoencoder.train(df)
        if basefolder:
            model, history, threshold, dir_name = autoencoder.retrain(df)
        else:
            # basemodel lub brak parametrów = train
            model, history, threshold, dir_name = autoencoder.train(df)

        response = {
                    "dir_name": dir_name,
                    "basemodel_used": basemodel if basemodel else None,
                    "basefolder_used": basefolder if basefolder else None
                }
        
        return JSONResponse(content = {'message': response},status_code = 200)
    except Exception as e:
        print(f"Error fetching performance metrics: {e}")
        return JSONResponse(
            content={'error': str(e)},
            status_code=500
        )

@app.on_event("startup")
async def on_startup():
    initialize_encoders()

@app.post("/save_settings")
async def save_settings(request: Request):
    # try:
    #     form_data = await request.form()
    #     settings = {
    #         "COLUMNS_TO_USE": form_data['columns'].split(','),
    #         "RELATIVE_FROM": int(datetime.strptime(form_data['relative_from'], '%Y-%m-%dT%H:%M').timestamp() * 1000),
    #         "RELATIVE_TO": int(datetime.strptime(form_data['relative_to'], '%Y-%m-%dT%H:%M').timestamp() * 1000),
    #         "epochs": int(form_data['epochs']),
    #         "batch_size": int(form_data['batch_size'])
    #     }

        return RedirectResponse(url="/", status_code=303)
    
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))

@app.get("/testing")
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})

@app.get("/model-results")
async def get_model_results(hids_folders: str = None, nids_folders: str = None):
    try:
        print("Fetching model results...")
        # Get all available folders
        all_hids_folders = get_all_folders(hids_encoder_base_path)
        all_nids_folders = get_all_folders(nids_encoder_base_path)
        
        # Parse input folders - comma separated list of folder names
        selected_hids_folders = []
        selected_nids_folders = []
        
        if hids_folders:
            selected_hids_folders = [folder.strip() for folder in hids_folders.split(',')]
            # Validate that all folders exist
            for folder in selected_hids_folders:
                if folder not in all_hids_folders:
                    return JSONResponse(
                        content={"error": f"HIDS folder '{folder}' not found"},
                        status_code=400
                    )
            # Convert to full paths
            selected_hids_folders = [os.path.join(hids_encoder_base_path, folder) for folder in selected_hids_folders]
        else:
            # Default to latest folder if none specified
            latest_hids = get_latest_folder(hids_encoder_base_path)
            if latest_hids:
                selected_hids_folders = [latest_hids]
        
        if nids_folders:
            selected_nids_folders = [folder.strip() for folder in nids_folders.split(',')]
            # Validate that all folders exist
            for folder in selected_nids_folders:
                if folder not in all_nids_folders:
                    return JSONResponse(
                        content={"error": f"NIDS folder '{folder}' not found"},
                        status_code=400
                    )
            # Convert to full paths
            selected_nids_folders = [os.path.join(nids_encoder_base_path, folder) for folder in selected_nids_folders]
        else:
            # Default to latest folder if none specified
            latest_nids = get_latest_folder(nids_encoder_base_path)
            if latest_nids:
                selected_nids_folders = [latest_nids]
        
        # Process HIDS data from multiple folders
        hids_data = process_multiple_folders(selected_hids_folders)
        
        # Process NIDS data from multiple folders
        nids_data = process_multiple_folders(selected_nids_folders)

        return JSONResponse(
            content={
                'hids': {
                    'folders': [os.path.basename(folder) for folder in selected_hids_folders],
                    'results': hids_data['results'],
                    'thresholds': hids_data['thresholds'],
                    'loss_history': hids_data['loss_history'],
                    'feature_importance': hids_data['feature_importance'],
                    'feature_distribution': hids_data['feature_distribution'],
                    'columns_used': hids_data['columns_used'],
                    'training_metadata': hids_data['training_metadata'],
                    'weights_heatmap_data': hids_data['weights_heatmap_data'],
                    'tsne_data': hids_data['tsne_data'],
                },
                'nids': {
                    'folders': [os.path.basename(folder) for folder in selected_nids_folders],
                    'results': nids_data['results'],
                    'thresholds': nids_data['thresholds'],
                    'loss_history': nids_data['loss_history'],
                    'feature_importance': nids_data['feature_importance'],
                    'feature_distribution': nids_data['feature_distribution'],
                    'columns_used': nids_data['columns_used'],
                    'training_metadata': nids_data['training_metadata'],
                    'weights_heatmap_data': nids_data['weights_heatmap_data'],
                    'tsne_data': nids_data['tsne_data'],
                },
                'available_folders': {
                    'hids': all_hids_folders,
                    'nids': all_nids_folders
                }
            }
        )
    except Exception as e:
        logging.error(f"Error fetching model results: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return JSONResponse(
            content={"error": f"Failed to fetch model results: {str(e)}"},
            status_code=500
        )

def process_multiple_folders(folders):
    aggregated_data = {
        'results': {},
        'thresholds': {},
        'loss_history': {},
        'feature_importance': {},
        'feature_distribution': {},
        'columns_used': set(),
        'training_metadata': {},
        'weights_heatmap_data': {},
        'tsne_data': {},
    }
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        
        # Load threshold value
        threshold_path = os.path.join(folder, 'threshold.json')
        threshold = None
        
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get("threshold", 0)
                aggregated_data['thresholds'][folder_name] = threshold

        weights_heatmap_path = os.path.join(folder, 'weights_heatmap_data.json')
        
        if os.path.exists(weights_heatmap_path):
            try:
                with open(weights_heatmap_path, 'r') as f:
                    weights_data = json.load(f)
                    aggregated_data['weights_heatmap_data'][folder_name] = weights_data
            except Exception as e:
                logging.error(f"Error loading weights heatmap data for {folder_name}: {str(e)}")

        # Load t-SNE data
        tsne_path = os.path.join(folder, 'tsne_data.json')
        
        if os.path.exists(tsne_path):
            try:
                with open(tsne_path, 'r') as f:
                    tsne_data = json.load(f)
                    aggregated_data['tsne_data'][folder_name] = tsne_data
            except Exception as e:
                logging.error(f"Error loading t-SNE data for {folder_name}: {str(e)}")
        
        # Load columns used in analysis
        columns_path = os.path.join(folder, 'columns_used.json')
        columns_used = []
        
        if os.path.exists(columns_path):
            with open(columns_path, 'r') as f:
                columns_used = json.load(f)
                aggregated_data['columns_used'].update(columns_used)
        
        # Load training metadata if available
        metadata_path = os.path.join(folder, 'training_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    aggregated_data['training_metadata'][folder_name] = metadata
            except Exception as e:
                logging.error(f"Error loading training metadata for {folder_name}: {str(e)}")     
        
        # Load anomaly detection results
        results_path = os.path.join(folder, 'anomaly_detection_results.csv')
        
        if os.path.exists(results_path):
            # Use na_filter=False to keep empty strings as empty strings
            # Set keep_default_na=False to prevent pandas from interpreting certain strings as NaN
            df = pd.read_csv(results_path, na_filter=False, keep_default_na=False)
            
            # Now replace any pandas NaN values with the string "NaN"
            df = df.fillna("NaN")
            
            # Apply threshold filter if available
            if threshold is not None:
                # Make sure mse is numeric for the comparison
                df["mse"] = pd.to_numeric(df["mse"], errors='coerce')
                df = df[df["mse"] > threshold]  # Keep only rows where mse > threshold
            
            # Debug information
            #print(f"Processing {folder_name}, found {len(df)} results after threshold filter")
            
            # Remove the source_folder column if it exists (from previously processed data)
            if 'source_folder' in df.columns:
                df = df.drop('source_folder', axis=1)
                
            # Instead of adding source_folder as a field, create a folder-based structure
            # Convert to records ensuring NaN becomes "NaN" string
            folder_results = df.to_dict(orient='records')
            
            # Debug to ensure we have data
            #print(f"Number of results for {folder_name}: {len(folder_results)}")
            
            # if len(folder_results) > 0:
            #     print(f"Sample result: {folder_results[0]}")
                
            aggregated_data['results'][folder_name] = folder_results
        
        # Load loss history data
        loss_path = os.path.join(folder, 'loss_history.csv')
        
        if os.path.exists(loss_path):
            loss_df = pd.read_csv(loss_path, na_filter=False, keep_default_na=False)
            loss_df = loss_df.fillna("NaN")
            folder_loss = loss_df.to_dict(orient='records')
            aggregated_data['loss_history'][folder_name] = folder_loss
        
        # Load feature importance data
        features_path = os.path.join(folder, 'feature_importance.csv')
        
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path, na_filter=False, keep_default_na=False)
            features_df = features_df.fillna("NaN")
            folder_features = features_df.to_dict(orient='records')
            aggregated_data['feature_importance'][folder_name] = folder_features
        
        # Load feature distribution data
        distribution_path = os.path.join(folder, 'anomaly_unique_values_summary.json')
        
        if os.path.exists(distribution_path):
            with open(distribution_path, 'r') as f:
                distribution_data = json.load(f)
                # Store distribution data by folder
                aggregated_data['feature_distribution'][folder_name] = distribution_data
    
    # Convert columns_used set back to list for JSON serialization
    aggregated_data['columns_used'] = list(aggregated_data['columns_used'])
    
    #print(f"Aggregated data: {aggregated_data['training_metadata']}")

    return aggregated_data
# Function to get all folders in a directory
def get_all_folders(base_path):
    try:
        if not os.path.exists(base_path):
            return []
        
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        return sorted(folders, reverse=True)  # Sort in descending order (newest first)
    except Exception as e:
        logging.error(f"Error getting folders: {str(e)}")
        return []

# Function to get the latest folder in a directory
def get_latest_folder(base_path):
    try:
        all_folders = get_all_folders(base_path)
        return os.path.join(base_path, all_folders[0]) if all_folders else None
    except Exception as e:
        logging.error(f"Error getting latest folder: {str(e)}")
        return None

@app.post("/processing")
async def processing_logs(request: Request):
    try:
        data = await request.json()
        nids_events = [item for item in data if item.get("event", {}).get("source") == "nids"]
        hids_events = [item for item in data if item.get("event", {}).get("source") == "hids"]
        
        if nids_events:
            #print("nids events: ", len(nids_events))

            df = pd.json_normalize(nids_events)
            columns_to_use = config.COLUMNS_TO_USE
            processed, processed_original = functions.preprocess(df, columns_to_use)

            encoder = EncoderInitializer("nids")
            scaler = encoder.scaler
            threshold = encoder.threshold
            autoencoder = encoder.autoencoder

            test_data_scaled = scaler.transform(processed)
            mse, anomalies = functions.detect_anomalies(autoencoder, test_data_scaled, threshold)

            os_actions = []
            current_date = datetime.now()
            index_formated_date = current_date.strftime("%Y.%m.%d")
            nids_index = config.NIDS_CONFIG["index"]
            index_name= f"{nids_index}-{index_formated_date}"
            for i, item in enumerate(nids_events):
                #print(anomalies[i])
                item.setdefault('anomaly_detection', {}).setdefault('is_anomaly', bool(anomalies[i]))
                item.setdefault('anomaly_detection', {}).setdefault('mse', float(mse[i]))
                item.setdefault('anomaly_detection', {}).setdefault('threshold', float(threshold))
                scoring = 0
                if mse[i] > 0:
                    scoring = (mse[i] - threshold) / mse[i] * 100
                
                item.setdefault('anomaly_detection', {}).setdefault('scoring', float(scoring))
                id_formated_date = current_date.strftime("%y%m%d")
                event_hash=item["event"]["hash"]
                new_document_id = f"{id_formated_date}-{event_hash}"

                action = {
                    "_index": index_name,
                    "_id": new_document_id,
                    "_source": item
                }


                os_actions.append(action)
            
            response = helpers.bulk(os_client, os_actions, max_retries=3)
            #print(response)
            send_to_webhook(os_actions)
            #print(nids_events[0]['anomaly_detection'])

        if hids_events:
            #print("hids events: ", len(hids_events))
            # print(hids_events[0])
            df = pd.json_normalize(hids_events)
            columns_to_use = config.COLUMNS_TO_USE_HIDS
            processed, processed_original = functions.preprocess(df, columns_to_use)

            encoder = EncoderInitializer("hids")
            scaler = encoder.scaler
            threshold = encoder.threshold
            autoencoder = encoder.autoencoder

            test_data_scaled = scaler.transform(processed)
            mse, anomalies = functions.detect_anomalies(autoencoder, test_data_scaled, threshold)

            os_actions = []
            current_date = datetime.now()
            index_formated_date = current_date.strftime("%Y.%m.%d")
            hids_index = config.HIDS_CONFIG["index"]
            index_name= f"{hids_index}-{index_formated_date}"
            for i, item in enumerate(hids_events):
                #print(anomalies[i])
                item.setdefault('anomaly_detection', {}).setdefault('is_anomaly', bool(anomalies[i]))
                item.setdefault('anomaly_detection', {}).setdefault('mse', float(mse[i]))
                item.setdefault('anomaly_detection', {}).setdefault('threshold', float(threshold))
                scoring = 0
                if mse[i] > 0:
                    scoring = (mse[i] - threshold) / mse[i] * 100
                
                item.setdefault('anomaly_detection', {}).setdefault('scoring', float(scoring))

                new_document_id = item["id"]

                action = {
                    "_index": index_name,
                    "_id": new_document_id,
                    "_source": item
                }

                os_actions.append(action)
            
            response = helpers.bulk(os_client, os_actions, max_retries=3)
            #print(response)
            send_to_webhook(os_actions)
            

    except Exception as e:
        print(f"Error processing data: {e}")
        return JSONResponse(content = {'message': f'Error processing data: {e}'},status_code = 500)

@app.get("/check_anomalies")
async def check_anomalies(request: Request):
    with encoder_lock:
        try:
            # Parse request parameters
            params = await request.json()
            event_source = params.get("event_source")
            event_data = params.get("data")

            print(f'event_source: {event_source}')

            # Determine which models to use based on event source
            if event_source == "nids":
                encoder = EncoderInitializer("nids")
                scaler = encoder.scaler
                threshold = encoder.threshold
                autoencoder = encoder.autoencoder
                columns_to_use = config.COLUMNS_TO_USE
            elif event_source == "hids":
                encoder = EncoderInitializer("hids")
                scaler = encoder.scaler
                threshold = encoder.threshold
                autoencoder = encoder.autoencoder
                columns_to_use = config.COLUMNS_TO_USE_HIDS
            else:
                return JSONResponse(
                    content={'message': 'Invalid event source'},
                    status_code=400
                )
                

            # Convert event data to DataFrame
            df = pd.json_normalize(event_data)

            # Preprocess the data
            processed, processed_original = functions.preprocess(df, columns_to_use)

            # Check if models are initialized
            if scaler is None or threshold is None or autoencoder is None:
                return JSONResponse(
                    content={
                        'scaler': None,
                        'threshold': None,
                        'autoencoder': None,
                        'message': 'Encoder components are not initialized'
                    },
                    status_code=200
                )

            # Scale the data
            test_data_scaled = scaler.transform(processed)

            # Detect anomalies
            mse, anomalies = functions.detect_anomalies(autoencoder, test_data_scaled, threshold)

            # Calculate scoring
            if mse[0] > 0:
                scoring = (mse[0] - threshold) / mse[0] * 100
            else:
                scoring = 0

            response_data = {
                'mse': mse[0],
                'threshold': threshold,
                'scoring': scoring,
                'is_anomaly': bool(anomalies)
            }

            print(response_data)
            return JSONResponse(
                content=response_data,
                status_code=200
            )

        except Exception as e:
            print(f"Error processing data: {e}")
            return JSONResponse(
                content={'message': f'Error processing data: {e}'},
                status_code=500
            )

def create_query_hids(params: Dict[str, Any]) -> Dict[str, Any]:
    if "normal_filters" in params:
        should_arr = []
        for nf in params["normal_filters"]:
            # Check if agent id is a list/array
            if isinstance(nf["agent"]["id"], (list, tuple)):
                # Create a terms query for multiple IDs
                query = {
                    "bool": {
                        "must": [
                            {
                                "terms": {
                                    "agent.id": nf["agent"]["id"]
                                }
                            }
                        ]
                    }
                }
            elif nf["agent"].get("id") == "*" or not nf["agent"].get("id"):
                # Handle wildcard or empty case - match all agents
                query = {
                    "bool": {
                        "must": [
                            {
                                "exists": {
                                    "field": "agent.id"
                                }
                            }
                        ]
                    }
                }
            else:
                # Handle single ID case (original behavior)
                query = {
                    "bool": {
                        "must": [
                            {
                                "match_phrase": {
                                    "agent.id": nf["agent"]["id"]
                                }
                            }
                        ]
                    }
                }
            should_arr.append(query)
    else:
        should_arr = [{"match_all": {}}]
    
    return {
        "size": 10000,
        "query": {
            "bool": {
                "filter": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": "now-1h/h",
                                        "lt": "now",
                                    }
                                },
                            }
                        ],
                        "should": should_arr,
                        "minimum_should_match": 1
                    }
                }
            }
        },
        "sort": [
            {
                "@timestamp": {
                    "order": "asc"
                }
            }
        ]
    }

def create_query_nids(params: dict):
    if "normal_filters" in params:
        should_arr = []
        for nf in params["normal_filters"]:
            query = {
                "bool": {
                    "must": [
                        {
                            "match_phrase": {
                                "source.ip": nf["source"]["ip"]
                            }
                        },
                        {
                            "range": {
                                "source.port": {
                                    "gte": nf["source"]["port"]["from"],
                                    "lte": nf["source"]["port"]["to"],
                                }
                            },
                        },
                        {
                            "range": {
                                "destination.port": {
                                    "gte": nf["destination"]["port"]["from"],
                                    "lte": nf["destination"]["port"]["to"],
                                }
                            },
                        },
                        {
                            "match_phrase": {
                                "destination.ip": nf["destination"]["ip"]
                            }
                        }
                    ]
                }
            }
            should_arr.append(query)
    else:
        should_arr = [{"match_all": {}}]
    
    return {
        "size": 10000,
        "query": {
            "bool": {
                "filter": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": "now-1h/h",
                                        "lt": "now",
                                    }
                                },
                            }
                        ],
                        "should": should_arr,
                        "minimum_should_match": 1
                    }
                }
            }
        },
        "sort": [
            {
                "@timestamp": {
                    "order": "asc"
                }
            }
        ]
    }

def get_latest_folder(base_path):
    #print(f"Searching for folders in: {base_path}")
    if not os.path.exists(base_path):
        raise ValueError(f"The base path does not exist: {base_path}")
    
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    #print(f"Found folders: {folders}")
    
    date_folders = []
    for folder in folders:
        try:
            datetime.strptime(folder, '%Y-%m-%d %H%M')
            date_folders.append(folder)
        except ValueError:
            print(f"Skipping non-date folder: {folder}")
    
    #print(f"Valid date folders: {date_folders}")
    
    if not date_folders:
        raise ValueError(f"No valid date folders found in {base_path}")
    
    latest_folder = max(date_folders, key=lambda x: datetime.strptime(x, '%Y-%m-%d %H%M'))
    return os.path.join(base_path, latest_folder)

uri = 'https://scp-anomalies.sec4b.corp/api/live-alerts/webhook/lckc_scp_test'
# ws = websocket.WebSocket()
def send_to_webhook(msg):
    # uri = uri + "?source=" + source
    r = requests.post(uri, json=msg, verify=False)
    # print(r)

def initialize_encoders():
    global nids_encoder, hids_encoder
    try:
        print("Initializing encoders...")
        nids_encoder = EncoderInitializer("nids")
        hids_encoder = EncoderInitializer("hids")
        print("Encoders initialized successfully.")
    except Exception as e:
        print(f"Error initializing encoders: {e}")
        nids_encoder = None
        hids_encoder = None
