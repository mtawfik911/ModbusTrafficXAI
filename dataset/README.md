# Modbus Traffic Dataset

This dataset contains packet-level data for Modbus network traffic. It is used for the classification of Modbus packets into normal and abnormal traffic. 

## Dataset Overview

- **Total Samples**: 6,690
- **Features**: 17 packet features (including padding) encoded in hexadecimal format.
- **Target**: The final column contains the class label (0 for normal, 1 for abnormal traffic).

## Dataset Structure

- **Columns**: 
  - **Feature Columns**: 17 columns representing packet data (hexadecimal).
  - **Target Column**: A binary label indicating whether the packet is normal (0) or abnormal (1).


## How to Use the Dataset

1. Download the dataset file: `modbus_traffic_data.csv`
2. Load the dataset in Python using libraries like `pandas`:

   ```python
   import pandas as pd
   data = pd.read_csv('modbus_traffic_data.csv')
