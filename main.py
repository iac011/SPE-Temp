import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import re

# ==========================================
# Directory Structure:
# /SPE_T_System/
#   ├── main.py                 (Main program)
#   ├── temperature_mapping.csv (Temperature mapping, contains 'index' and 'temp' columns)
#   ├── /database/              (Training spectra database)
#   │   ├── sample_1.txt
#   │   ├── sample_2.txt
#   │   ├── ...
#   │   ├── sample_1800.txt
#   ├── /to_be_measured/        (Unknown spectra to be determined)
#       ├── any_name_A.txt
#       ├── any_name_B.txt
# ==========================================

# 1. Define AI Model (SPE-T 1D-CNN)
class SPET_Net(nn.Module):
    def __init__(self, input_dim):
        super(SPET_Net, self).__init__()
        # Use 1D Convolutional Neural Network (1D-CNN) to extract local spectral features (e.g., peak width, peak shift)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16) # Adaptive pooling, outputs fixed length regardless of input dimension
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output layer: Determined temperature
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1) # Add channel dimension: (batch_size, 1, input_dim)
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.regressor(x)
        return x

# 1.5 Data Normalization (Sample-wise Normalization)
def normalize_spectra(spectra):
    """
    Sample-wise Normalization
    Eliminates the impact of absolute intensity fluctuations (e.g., laser power, sample concentration),
    forcing the model to focus only on the spectral "shape" (peak position, linewidth), which is key for temperature determination.
    """
    norm_spectra = np.zeros_like(spectra, dtype=np.float32)
    for i in range(len(spectra)):
        s_min = np.min(spectra[i])
        s_max = np.max(spectra[i])
        if s_max > s_min:
            norm_spectra[i] = (spectra[i] - s_min) / (s_max - s_min)
        else:
            norm_spectra[i] = spectra[i]
    return norm_spectra

# 2. Load Temperature Mapping
def load_temperature_mapping():
    """
    Read temperature mapping from the attachment.
    Assumes the attachment is saved as temperature_mapping.csv with 'index' and 'temp' columns.
    """
    mapping = {}
    try:
        # In a real scenario, uncomment the code below to read your actual data:
        # df_map = pd.read_csv('temperature_mapping.csv')
        # for _, row in df_map.iterrows():
        #     mapping[int(row['index'])] = float(row['temp'])
        
        # --- Simulated data for demonstration (1-1800 maps to 20K-450K) ---
        print("Note: Using simulated temperature mapping. Please replace with your actual data!")
        temps = np.linspace(20, 450, 1800)
        for i in range(1, 1801):
            mapping[i] = temps[i-1]
        # ---------------------------------------------------
    except Exception as e:
        print(f"Failed to load temperature mapping: {e}")
    return mapping

# 3. Data Loading Module
def load_database_spectra(folder_path, temp_mapping, max_files=1700):
    """
    Read spectra from the database folder.
    Naming convention: ?_1.txt, ?_2.txt ... ?_1800.txt
    Only read the first max_files (1700) files.
    """
    spectra = []
    temperatures = []
    
    # Match files ending with _number.txt
    file_pattern = os.path.join(folder_path, '*_[0-9]*.txt')
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"Warning: No files found in {folder_path}")
        return np.array([]), np.array([])
        
    for file in files:
        filename = os.path.basename(file)
        # Extract the numeric index after the underscore using regex (e.g., sample_150.txt -> 150)
        match = re.search(r'_(\d+)\.txt$', filename)
        if match:
            idx = int(match.group(1))
            
            # Limit to reading only the first 1700 files
            if idx > max_files:
                continue
                
            try:
                # Read spectral data (3 columns: Wavelength, Separator, Intensity)
                df = pd.read_csv(file, sep='\s+', header=None)
                # Extract the 3rd column (index 2) as the intensity feature vector
                intensity_vector = df.iloc[:, 2].values
                
                spectra.append(intensity_vector)
                temperatures.append(temp_mapping[idx])
            except Exception as e:
                print(f"Error reading {file}: {e}")
            
    return np.array(spectra), np.array(temperatures)

def load_unknown_spectra(folder_path):
    """Read all spectral files in the to_be_measured folder (any filename)"""
    spectra = []
    file_names = []
    
    # Match all txt and csv files, regardless of filename
    files = glob.glob(os.path.join(folder_path, '*.txt')) + glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file in files:
        try:
            # Compatible with txt (space-separated) or csv (comma-separated)
            sep = ',' if file.endswith('.csv') else '\s+'
            # Read spectral data (3 columns: Wavelength, Separator, Intensity)
            df = pd.read_csv(file, sep=sep, header=None)
            # Extract the 3rd column (index 2) as the intensity feature vector
            intensity_vector = df.iloc[:, 2].values
            spectra.append(intensity_vector)
            file_names.append(os.path.basename(file))
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    return np.array(spectra), file_names

# 4. Main Program
def main():
    DB_FOLDER = './database/'
    MEASURE_FOLDER = './to_be_measured/'
    
    # --- Phase A: Model Training ---
    print(">>> Phase A: Loading database and training model")
    
    # 1. Load temperature mapping
    temp_mapping = load_temperature_mapping()
    
    # 2. Load the first 1700 training data files
    X_train, y_train = load_database_spectra(DB_FOLDER, temp_mapping, max_files=1700)
    
    if len(X_train) == 0:
        print("Error: Database is empty or no files matching the naming convention were found.")
        return
        
    print(f"Successfully loaded {len(X_train)} training spectra (indices 1-1700).")
    y_train = y_train.reshape(-1, 1)
    
    # 3. Data Normalization (Sample-wise normalization to extract line-shape features)
    X_train_scaled = normalize_spectra(X_train)
    
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train)
    
    # 4. Initialize model
    input_dimension = X_train.shape[1]
    model = SPET_Net(input_dim=input_dimension)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training loop
    epochs = 500
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print("Model training completed!\n")
    
    # --- Phase B: Unknown Spectra Determination ---
    print(">>> Phase B: Reverse determination of unknown spectra temperature")
    
    X_unknown, unknown_filenames = load_unknown_spectra(MEASURE_FOLDER)
    
    if len(X_unknown) == 0:
        print("Note: No spectra to be measured found. Please place files in the ./to_be_measured/ folder.")
        return
        
    # Must use the same normalization method
    X_unknown_scaled = normalize_spectra(X_unknown)
    X_unknown_t = torch.FloatTensor(X_unknown_scaled)
    
    # AI Determination
    model.eval()
    with torch.no_grad():
        predicted_temps = model(X_unknown_t).numpy()
        
    print("\n--- Determination Results ---")
    for i, filename in enumerate(unknown_filenames):
        pred_t = predicted_temps[i][0]
        print(f"File: {filename}  =>  AI Determined Temperature: {pred_t:.2f} K")

if __name__ == "__main__":
    main()
