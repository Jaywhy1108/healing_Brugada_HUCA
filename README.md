# README: Healing Brugada-HUCA

This guide outlines the setup, configuration, and execution steps for running the **Healing Brugada-HUCA** pipeline within a Google Colab environment.

## 1. Environment Setup and Prerequisites

To run this pipeline in Google Colab, you must first connect your Google Drive and install the required dependencies.

**Mounting Google Drive**
The script requires access to your Google Drive to load the dataset and metadata. Ensure your Colab notebook mounts the drive first:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
*Note: Your working directory should be set to `/content`.*

**Installing Dependencies**
You must install specific libraries for clinical electrocardiography processing and model interpretability. Run the following cell at the beginning of your notebook:
```python
pip install wfdb neurokit2 tensorflow tensorflow.keras
```
The script also requires `pywt` (PyWavelets) for Continuous Wavelet Transforms, `cv2` (OpenCV) for image resizing, `shap` for model interpretability, and standard scientific libraries like `scipy`, `sklearn`, `pandas`, and `matplotlib`. 

## 2. Directory Structure and Configuration

Ensure your Google Drive is structured so the script can locate the raw data. By default, the configuration is set to:
*   **DATA_DIR**: `/content/drive/MyDrive/Colab Notebooks/brugada/files` (Contains the patient folders with multi-lead ECGs).
*   **METADATA_PATH**: `/content/drive/MyDrive/Colab Notebooks/brugada/metadata.csv`.

**Global Configuration Variables**
The model targets leads **V1, V2, and V3** and natively processes at **100Hz** to prevent interpolation artifacts. The `LABEL_MODE` is set to "inclusive", marking any patient with a Brugada value $\ge 1$ as positive. 

## 3. Execution Pipeline

The execution of the code is broken down into several distinct phases. It is recommended to run the cells sequentially.

### Phase 1: Signal Loading, Preprocessing, and Segmentation
1.  **Filtering**: The pipeline applies a 50Hz Notch filter to remove electrical noise and a 0.5 to 40Hz Bandpass filter to remove baseline wander.
2.  **R-Peak Detection & Segmentation**: Using `neurokit2`, the code isolates individual heartbeats by extracting a window from -0.25 seconds to +0.45 seconds around the detected R-peak.
3.  **Artifact Rejection**: Premature beats, flatlines, and extreme noise are automatically rejected during this phase.

### Phase 2: Custom SE-ResNet Modeling (1D Data)
1.  **Data Splitting**: Data is split at the *patient level* (Train/Validation/Test) to prevent data leakage.
2.  **Training**: A custom 1D SE-ResNet model is trained using Focal Loss (to handle severe dataset class imbalance).
3.  **Evaluation**: The model automatically tunes its decision threshold using the Precision-Recall curve to optimize the F1-score before rendering patient-level clinical evaluations and ROC curves.

### Phase 3: 2D CWT Spectrogram Conversion
To utilize state-of-the-art vision models, the 1D heartbeat time-series data is converted into 2D RGB images:
1.  A **Continuous Wavelet Transform (CWT)** using the Morlet wavelet translates the electrical frequencies into a visual spectrum.
2.  Leads V1, V2, and V3 are mapped to the Red, Green, and Blue channels, respectively.
3.  **Memory Warning**: The scalograms are resized to 224x224 and cast to `np.uint8` to drastically save Colab RAM (using 1 Byte per pixel instead of 4 or 8). 

### Phase 4: ResNet-50 Transfer Learning
1.  **Formatting**: The images are passed through ResNet's `preprocess_input` to convert them to the expected zero-centered, BGR format.
2.  **Base Training**: A pre-trained ResNet50 model is loaded with its base layers **frozen** (`base_model.trainable = False`). A custom classification head with dropout layers is trained on top.

### Phase 5: ResNet-50 Fine-Tuning
1.  **Unfreezing**: The top 30 layers of the ResNet50 base model are unfrozen to allow the network to learn specific Brugada shapes.
2.  **Microscopic Learning Rate**: The fine-tuning is compiled with a very small learning rate (`1e-5`) to avoid destroying the pre-trained ImageNet weights.

### Phase 6: Model Interpretability (Deep SHAP)
To ensure the model is clinically sound and not acting as a "black box":
1.  A `shap.DeepExplainer` is built using a background dataset from the test set.
2.  The script calculates SHAP values and segments the attention over specific ECG regions (e.g., P-wave, PR-segment, QRS-complex, ST-segment, T-wave).
3.  The model's decision-making is summarized; for instance, ideally placing the majority of its focus on the **ST-segment** and **T-wave** (repolarization pattern) to confirm it is organically identifying the Brugada anomaly.
