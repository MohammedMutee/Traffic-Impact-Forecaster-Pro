
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Define Target: Impact Score
    # 0: Low, 1: Medium, 2: High, 3: Extreme
    def calculate_impact(row):
        score = 0
        is_high_priority = row['priority'] == 'High'
        closure = row['requires_road_closure']
        critical_cause = row['event_cause'] in ['public_event', 'protest', 'procession', 'vip_movement']
        
        if (row['event_type'] == 'planned') or critical_cause or (closure and is_high_priority):
            return 3
        elif is_high_priority or row['event_cause'] == 'accident':
            return 2
        elif closure:
            return 1
        else:
            return 0

    df['impact_score'] = df.apply(calculate_impact, axis=1)
    
    # 2. Temporal Features
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df = df.dropna(subset=['start_datetime'])
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['start_datetime'].dt.month
    
    # 3. Categorical Features
    cat_cols = ['event_type', 'event_cause', 'corridor', 'police_station', 'veh_type', 'zone']
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        
    # 4. Spatial Features (handling coordinates)
    df['latitude'] = df['latitude'].replace(0, np.nan)
    df['longitude'] = df['longitude'].replace(0, np.nan)
    df['latitude'] = df['latitude'].fillna(df['latitude'].median())
    df['longitude'] = df['longitude'].fillna(df['longitude'].median())
    
    # Select features
    features = ['hour', 'day_of_week', 'is_weekend', 'month', 'latitude', 'longitude'] + \
               [col + '_enc' for col in cat_cols]
    
    X = df[features]
    y = df['impact_score']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("\nImpact Score Distribution in Train:")
    print(y_train.value_counts(normalize=True))
