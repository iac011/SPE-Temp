# SPE-T (Spectral-Temperature) 1D-CNN Framework

A deep learning framework designed to determine temperature from Photoluminescence (PL) spectra using a 1D Convolutional Neural Network (1D-CNN). 

Unlike traditional ratiometric methods (FIR) that rely on a single intensity ratio, SPE-T monitors the entire high-dimensional spectral line-shape. It employs sample-wise normalization and convolutional layers to capture local spectral features such as peak broadening and peak shifts, making it highly robust against absolute intensity fluctuations caused by laser power variations or sample concentration.

## Features

- **1D-CNN Architecture**: Acts as a digital filter to extract local spectral features (peak shifts, broadening) rather than treating wavelengths as isolated data points.
- **Sample-wise Normalization**: Eliminates the impact of absolute intensity fluctuations, forcing the model to focus strictly on the spectral "shape".
- **Automated Batch Processing**: Automatically scans directories, matches filenames, and processes large datasets of spectra.
- **Flexible Input Handling**: Supports both `.txt` and `.csv` files, specifically handling 3-column data formats (Wavelength, Separator, Intensity).

## Directory Structure

To use this program, ensure your project directory is structured as follows:

```text
/SPE_T_System/
  ├── main.py                 (The main Python script)
  ├── temperature_mapping.csv (Temperature mapping file)
  ├── /database/              (Training spectra database)
  │   ├── sample_1.txt
  │   ├── sample_2.txt
  │   ├── ...
  │   ├── sample_1800.txt
  ├── /to_be_measured/        (Unknown spectra to be determined)
      ├── unknown_A.txt
      ├── unknown_B.txt
```

## Data Format

### Spectral Files (`.txt` or `.csv`)
The program expects the spectral data files to contain 3 columns (separated by spaces, tabs, or commas):
1. **Wavelength** (nm)
2. **Separator** (e.g., a numeric separator, ignored by the model)
3. **Luminescence Intensity** (counts)

*Note: The model automatically extracts the 3rd column (index 2) as the intensity feature vector.*

### Temperature Mapping (`temperature_mapping.csv`)
For the training phase, the program needs to know the temperature corresponding to each file in the `database` folder. 
Create a `temperature_mapping.csv` file with the following columns:
- `index`: The numeric index matching the filename (e.g., `1` for `sample_1.txt`).
- `temp`: The corresponding temperature in Kelvin (K).

*(Note: The provided `main.py` contains a simulated mapping for demonstration purposes. Uncomment the CSV reading section in `load_temperature_mapping()` to use your actual file).*

## Usage

1. **Install Dependencies**:
   Ensure you have the required Python libraries installed:
   ```bash
   pip install numpy pandas torch
   ```

2. **Prepare Data**:
   - Place your training spectra in the `/database/` folder. Ensure they follow the naming convention `*_1.txt`, `*_2.txt`, etc.
   - Place your unknown spectra in the `/to_be_measured/` folder. They can have any filename.
   - Provide your `temperature_mapping.csv` in the root directory.

3. **Run the Program**:
   Execute the main script:
   ```bash
   python main.py
   ```

## Workflow

1. **Phase A: Model Training**
   - Loads the temperature mapping.
   - Reads the first 1700 files from the `/database/` folder.
   - Applies sample-wise normalization to the intensity vectors.
   - Trains the 1D-CNN model (`SPET_Net`) using Mean Squared Error (MSE) loss and the Adam optimizer for 500 epochs.

2. **Phase B: Unknown Spectra Determination**
   - Reads all `.txt` and `.csv` files from the `/to_be_measured/` folder.
   - Applies the exact same sample-wise normalization.
   - Uses the trained 1D-CNN model to determine the temperature.
   - Outputs the determined temperature for each file to the console.

## Citation

If you use this code or framework in your research or project, please cite it as follows:

**BibTeX:**
```bibtex
@software{spet_1dcnn_2026,
  author = {Guanyu Cai},
  title = {SPE-T: A 1D-CNN Framework for Spectral-Temperature Determination},
  year = {2026},
  publisher = {GitHub},
  journal = {SPE-Temp},
  howpublished = {\\url{https://github.com/iac011/SPE-Temp}}
}
```

**APA:**
> Guanyu Cai. (2026). *SPE-T: A 1D-CNN Framework for Spectral-Temperature Determination* [Computer software]. GitHub. https://github.com/iac011/SPE-Temp

## License
MIT License
