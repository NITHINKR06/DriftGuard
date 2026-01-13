# Save Feature Pipeline

from sklearn.preprocessing import StandardScaler

def scale_features(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler
