from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import json 
import csv
import os
import pandas as pd
from fractions import Fraction
from datetime import datetime
import sys
from opensearchpy import OpenSearch

sys.path.append('/app/config')
from config import functions, config
import requests

# Import all relevant classes
sys.path.append('/app/retrain/autoencoder_wazuh_classes')
from autoencoder_wazuh_classes import (
    AutoencoderSystem, DataProcessor, AutoencoderModel, 
    AnomalyDetector, FeatureImportance, ModelManager, 
    check_encoder_wazuh_folder
)

sys.path.append('/app/retrain/autoencoder_with_settings_retrain_classes')
from autoencoder_with_settings_retrain_classes import (
    AutoencoderSystem, DataProcessor, AutoencoderModel, 
    AnomalyDetector, FeatureImportance, ModelManager
)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the AutoencoderSystem
autoencoder_system = AutoencoderSystem()
data_processor = DataProcessor()
model_manager = ModelManager()

# Pydantic models for request validation
class Agent(BaseModel):
    id: Union[str, List[str]]

class NormalFilterWazuh(BaseModel):
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

class TrainRequestWazuh(BaseModel):
    normal_filters: List[NormalFilterWazuh]

class TrainRequestNids(BaseModel):
    normal_filters: List[NormalFilterNids]

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/save_settings")
async def save_settings(request: Request):
    try:
        form_data = await request.form()
        settings = {
            "COLUMNS_TO_USE": form_data['columns'].split(','),
            "RELATIVE_FROM": int(datetime.strptime(form_data['relative_from'], '%Y-%m-%dT%H:%M').timestamp() * 1000),
            "RELATIVE_TO": int(datetime.strptime(form_data['relative_to'], '%Y-%m-%dT%H:%M').timestamp() * 1000),
            "epochs": int(form_data['epochs']),
            "batch_size": int(form_data['batch_size'])
        }

        return RedirectResponse(url="/", status_code=303)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/testing")
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})

@app.post("/train/wazuh")
async def train(request: TrainRequestWazuh):
    print("training process start")
    normal_filters = request.normal_filters

    query = create_query_wazuh({"normal_filters": [filter.dict() for filter in normal_filters]})
    print(query)
    df = functions.get_opensearch_data(config.WAZUH_CONFIG, query, config.COLUMNS_TO_USE_WAZUH)
    print(df)

    # Check encoder_wazuh folder
    encoder_path = 'encoder_wazuh'
    
    # First check if folder exists, if not create it
    if not os.path.exists(encoder_path):
        os.makedirs(encoder_path)
        print("Created encoder_wazuh folder")
        print("Initial training started")
        model, history, threshold = autoencoder_system.train(df)
    else:
        # Check if folder has any contents (looking for model files)
        folder_contents = os.listdir(encoder_path)
        if not folder_contents:  # folder is empty
            print("encoder_wazuh folder is empty")
            print("Initial training started")
            model, history, threshold = autoencoder_system.train(df)
        else:
            # Check if any subfolder contains model files
            has_model = False
            for item in folder_contents:
                subfolder_path = os.path.join(encoder_path, item)
                if os.path.isdir(subfolder_path):
                    # Check for model files in subfolder
                    if os.path.exists(os.path.join(subfolder_path, 'autoencoder.h5')):
                        has_model = True
                        break
            
            if has_model:
                print("Found existing model")
                print("Retraining started")
                model, history, threshold = autoencoder_system.retrain(df)
            else:
                print("No valid model found in encoder_wazuh folder")
                print("Initial training started")
                model, history, threshold = autoencoder_system.train(df)

    return {
        "message": "Training completed",
        "folder_status": "Model saved in encoder_wazuh folder"
    }

@app.post("/train/nids")
async def train(request: TrainRequestNids):
    print("training process start")
    normal_filters = request.normal_filters

    query = create_query_nids({"normal_filters": [filter.dict() for filter in normal_filters]})
    print(query)
    df = functions.get_opensearch_data(config.NIDS_CONFIG, query, config.COLUMNS_TO_USE)
    print(df)

    # Check encoder folder
    encoder_path = 'encoder'
    
    # First check if folder exists, if not create it
    if not os.path.exists(encoder_path):
        os.makedirs(encoder_path)
        print("Created encoder folder")
        print("Initial training started")
        model, history, threshold = autoencoder_system.train(df)
    else:
        # Check if folder has any contents (looking for model files)
        folder_contents = os.listdir(encoder_path)
        if not folder_contents:  # folder is empty
            print("encoder folder is empty")
            print("Initial training started")
            model, history, threshold = autoencoder_system.train(df)
        else:
            # Check if any subfolder contains model files
            has_model = False
            for item in folder_contents:
                subfolder_path = os.path.join(encoder_path, item)
                if os.path.isdir(subfolder_path):
                    # Check for model files in subfolder
                    if os.path.exists(os.path.join(subfolder_path, 'autoencoder.h5')):
                        has_model = True
                        break
            
            if has_model:
                print("Found existing model")
                print("Retraining started")
                model, history, threshold = autoencoder_system.retrain(df)
            else:
                print("No valid model found in encoder folder")
                print("Initial training started")
                model, history, threshold = autoencoder_system.train(df)

    return {
        "message": "Training completed",
        "folder_status": "Model saved in encoder folder"
    }

def create_query_wazuh(params: Dict[str, Any]) -> Dict[str, Any]:
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
                                        "gte": "now-96h/h",
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

def create_query_nids(params: Dict[str, Any]) -> Dict[str, Any]:
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
                                        "gte": "now-7d/d",
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='172.19.20.25', port=8001)