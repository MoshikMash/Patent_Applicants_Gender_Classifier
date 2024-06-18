import json
from model import train
from preprocessing import preprocessing

config_path = "../../configs/base_config.json"
with open(config_path) as f:
    config = json.load(f)
# preprocessing(config)
train(config)
