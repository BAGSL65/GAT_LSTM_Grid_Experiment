import os
import logging
import pandas as pd
import numpy as np
import random
import torch
import networkx as nx
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data
import matplotlib.pyplot as plt
# 设置随机种子
seed = 65
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set up logging
logging.basicConfig(level=logging.INFO)

# 获取执行文件的绝对路径
script_path = os.path.abspath(__file__)

# 获取执行文件所在的目录
script_dir = os.path.dirname(script_path)

def load_config():
    """Load configuration settings."""
    return {
        'dynamic_data_path': os.path.join(script_dir, 'my_data/dynamic_data.csv'),
        'static_data_path': os.path.join(script_dir, 'my_data/static_data.csv'),
        'grid_data_path': os.path.join(script_dir, 'my_data/grid_df.csv'),
        'sequence_length': 24,
        'train_start_date': '2019-01-01',
        'train_end_date': '2019-12-31',
        'val_start_date': '2020-01-01',
        'val_end_date': '2020-06-30',
        'test_start_date': '2020-07-01',
        'output_dir': os.path.join(script_dir, 'outputs/Transformer_Plots')
    }


def save_processed_data(output_dir, **data):
    """Save processed data to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save tensors
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            file_path = os.path.join(output_dir, f"{key}.pt")
            torch.save(value, file_path)
            print(f"Saved {key} to {file_path}")
        elif isinstance(value, RobustScaler):
            file_path = os.path.join(output_dir, f"{key}.pkl")
            with open(file_path, 'wb') as f:
                import pickle
                pickle.dump(value, f)
            print(f"Saved {key} to {file_path}")


def load_and_prepare_data(config):
    """Load and prepare datasets."""
    dynamic_data = pd.read_csv(config['dynamic_data_path'])
    static_data = pd.read_csv(config['static_data_path'])
    grid_df = pd.read_csv(config['grid_data_path'])

    # Convert datetime and sort data
    dynamic_data['datetime'] = pd.to_datetime(dynamic_data['datetime'])
    dynamic_data = dynamic_data.sort_values(by=['state', 'datetime'])
    dynamic_data['target_consumption'] = dynamic_data.groupby('state')['value'].shift(-1)
    dynamic_data = dynamic_data.dropna(subset=['target_consumption'])

    return dynamic_data, static_data, grid_df

def split_data(dynamic_data, config):
    """Split data into train, validation, and test sets."""
    train_data = dynamic_data[(dynamic_data['datetime'] >= config['train_start_date']) & (dynamic_data['datetime'] < config['train_end_date'])].copy()
    val_data = dynamic_data[(dynamic_data['datetime'] >= config['val_start_date']) & (dynamic_data['datetime'] < config['val_end_date'])].copy()
    test_data = dynamic_data[dynamic_data['datetime'] >= config['test_start_date']].copy()

    for data in [train_data, val_data, test_data]:
        data['time_index'] = (data['datetime'] - data['datetime'].min()).dt.total_seconds() // 3600
        data['time_index'] = data['time_index'].astype(int)

    return train_data, val_data, test_data

def scale_features(train_data, val_data, test_data, features_to_scale):
    """Scale features using RobustScaler."""
    feature_scaler = RobustScaler()
    train_data[features_to_scale] = feature_scaler.fit_transform(train_data[features_to_scale])
    val_data[features_to_scale] = feature_scaler.transform(val_data[features_to_scale])
    test_data[features_to_scale] = feature_scaler.transform(test_data[features_to_scale])
    return feature_scaler

def scale_targets(train_data, val_data, test_data):
    """Scale target consumption using RobustScaler."""
    target_scaler = RobustScaler()
    train_data['target_consumption'] = target_scaler.fit_transform(train_data[['target_consumption']])
    val_data['target_consumption'] = target_scaler.transform(val_data[['target_consumption']])
    test_data['target_consumption'] = target_scaler.transform(test_data[['target_consumption']])
    return target_scaler

def create_sequences(data, sequence_length, features_to_scale):
    """Create input sequences, targets, and node indices."""
    sequences, targets, nodes = [], [], []
    grouped = data.groupby('state')

    for state, group in grouped:
        group = group.sort_values(by='datetime')
        values = group[features_to_scale].values
        target_values = group['target_consumption'].values

        if len(group) >= sequence_length + 1:
            for i in range(len(group) - sequence_length):
                seq = values[i:i + sequence_length]
                tgt = target_values[i + sequence_length]
                sequences.append(seq)
                targets.append(tgt)

    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float).unsqueeze(-1)
    nodes_tensor = torch.tensor(np.array(nodes), dtype=torch.long)

    return sequences_tensor, targets_tensor, nodes_tensor


# Main preprocessing function
def preprocess_data(config):
    dynamic_data, static_data, grid_df = load_and_prepare_data(config)
    train_data, val_data, test_data = split_data(dynamic_data, config)

    features_to_scale = ['value', 'pv', 'onw', 'ofw',
    'TOTAL HOURLY RAIN (mm)(mean)', 'TOTAL HOURLY RAIN (mm)(std)',
    'ATMOSPHERIC PRESSURE AT STATION LEVEL (mB)(mean)',
    'ATMOSPHERIC PRESSURE AT STATION LEVEL (mB)(std)',
    'GLOBAL RADIATION (KJ/m²)(mean)', 'GLOBAL RADIATION (KJ/m²)(std)',
    'AIR TEMPERATURE (°C)(mean)', 'AIR TEMPERATURE (°C)(std)',
    'DEW POINT TEMPERATURE (°C)(mean)', 'DEW POINT TEMPERATURE (°C)(std)',
    'REL HUMIDITY FOR THE LAST HOUR (%)(mean)',
    'REL HUMIDITY FOR THE LAST HOUR (%)(std)', 'WIND DIRECTION (gr)(mean)',
    'WIND DIRECTION (gr)(std)', 'WIND MAXIMUM GUST (m/s)(mean)',
    'WIND MAXIMUM GUST (m/s)(std)', 'WIND SPEED (m/s)(mean)',
    'WIND SPEED (m/s)(std)','year', 'month', 'day', 'hour', 'dayofweek',
    'weekofyear', 'quarter', 'is_holiday', 'season', 'state_id',
    'total_plant_capacity', 'population', 'GDP']  # Add your features list here 'population', 'GDP', 'year'
    
    feature_scaler = scale_features(train_data, val_data, test_data, features_to_scale)
    target_scaler = scale_targets(train_data, val_data, test_data)

    train_seq, train_tgt, train_nodes = create_sequences(train_data, config['sequence_length'], features_to_scale)
    val_seq, val_tgt, val_nodes = create_sequences(val_data, config['sequence_length'], features_to_scale)
    test_seq, test_tgt, test_nodes = create_sequences(test_data, config['sequence_length'], features_to_scale)

    # Save processed outputs
    save_processed_data(
        config['output_dir'],
        train_seq=train_seq,
        train_tgt=train_tgt,
        train_nodes=train_nodes,
        val_seq=val_seq,
        val_tgt=val_tgt,
        val_nodes=val_nodes,
        test_seq=test_seq,
        test_tgt=test_tgt,
        test_nodes=test_nodes,
        target_scaler=target_scaler
    )

    return train_seq, train_tgt, val_seq, val_tgt, test_seq, test_tgt, target_scaler

