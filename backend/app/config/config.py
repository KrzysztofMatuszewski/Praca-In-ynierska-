from dotenv import load_dotenv
import os

class Config:

    def __init__(self):
        load_dotenv()
        # for key, value in os.environ.items():
        #     setattr(self, key, convert_value(value))
        self.COLUMNS_TO_USE = os.getenv('COLUMNS_TO_USE', '').split(',')
        self.COLUMNS_TO_USE_HIDS = os.getenv('COLUMNS_TO_USE_WAZUH', '').split(',')

        self.RELATIVE_FROM = os.getenv('RELATIVE_FROM', '')
        self.RELATIVE_TO = os.getenv('RELATIVE_TO', '')

        self.TEST_SET_SIZE = int(os.getenv('TEST_SET_SIZE', 1000))
        self.EPOCHS = int(os.getenv('EPOCHS', 50))
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))

        self.HIDS_CONFIG = {
            "user": os.getenv('HIDS_USER'),
            "password": os.getenv('HIDS_PASSWORD'),
            "host": os.getenv('HIDS_HOST'),
            "port": int(os.getenv('HIDS_PORT', 9200)),
            "index": os.getenv('HIDS_INDEX')
        }

        self.NIDS_CONFIG = {
            "user": os.getenv('NIDS_USER'),
            "password": os.getenv('NIDS_PASSWORD'),
            "host": os.getenv('NIDS_HOST'),
            "port": int(os.getenv('NIDS_PORT', 9200)),
            "index": os.getenv('NIDS_INDEX')
        }

        self.HIDS_TRAIN_RAW_DATA = "app/data/hids/TRAIN_RAW_DATA.csv"
        self.HIDS_TRAIN_DATA = "app/data/hids/TRAIN_DATA.csv"
        self.NIDS_TRAIN_RAW_DATA = "app/data/nids/TRAIN_RAW_DATA.csv"
        self.NIDS_TRAIN_DATA = "app/data/nids/TRAIN_DATA.csv"


