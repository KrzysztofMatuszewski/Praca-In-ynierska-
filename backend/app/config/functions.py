import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
from opensearchpy import OpenSearch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


# TODO - Poprawić query aby zaczytywać tylko z columns zamist obcinac już pobrane o te kolumny

def get_opensearch_data(config, query, columns):
    auth = (config["user"], config["password"])
    client = OpenSearch(
        hosts=[{'host': config["host"], 'port': config["port"]}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=1000
    )

    SIZE = 10000

    first_page = client.search(
        body=query,
        index=config["index"],
        scroll='1m',
        timeout=1000
    )
    scroll_id = first_page['_scroll_id']
    new_page = first_page
    all_hits = []
    all_hits = all_hits + new_page['hits']['hits']
    PAGE_NUM = 1

    while True:
        if len(new_page['hits']['hits']) < SIZE:
            break
        PAGE_NUM += 1
        new_page = client.scroll(
            scroll_id=scroll_id,
            scroll='1m'
        )
        scroll_id = new_page['_scroll_id']
        hits = new_page['hits']['hits']
        all_hits += hits

    print(f'pages: {PAGE_NUM} len: {len(all_hits)}')

    # Inicjalizacja pustej listy przed pętlą
    extracted_data = []

    for document in all_hits:
        extracted_variables = {}

        for var_name in columns:
            keys = var_name.split('.')
            # Przechodzimy przez ścieżkę kluczy, aby uzyskać wartość
            value = document["_source"]
            for key in keys:
                value = value.get(key, None)
                if value is None:
                    break

            # Przypisujemy uzyskaną wartość do odpowiedniego klucza w słowniku `extracted_variables`
            extracted_variables[var_name] = value

        # Dodajemy wyodrębnione zmienne do listy
        extracted_data.append(extracted_variables)

    # Tworzenie DataFrame z wyodrębnionych danych
    df = pd.DataFrame(extracted_data)
    print(df)
    return df

def preprocess(inputDf, columns_to_use):
    print(columns_to_use)
    df = inputDf.loc[:, columns_to_use]
    df_original = df.copy()

    for column in columns_to_use:
        # Konwersja list na krotki przed haszowaniem
        print(column)
        if column == 'source.port' or column == 'destination.port':
            df[column] = pd.to_numeric(df[column])
        df[column] = df[column].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        df[column] = pd.util.hash_pandas_object(df[column], index=False, hash_key="95f9a5658c386e04")
    
    return df, df_original

def detect_anomalies(model, data, threshold):
    reconstructed = model.predict(data)
    mse = np.mean(np.power(data - reconstructed, 2), axis=1)
    # print(mse.shape)
    # print(f'mse: {data}')
    # print(f'mse: {reconstructed}')
    # print(f'mse: {mse}')
    # print(f'threshold: {threshold}')

    return mse, mse > threshold