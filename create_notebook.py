import json
import re

# Read the working Python file
with open('ECG_Pipeline.py', 'r') as f:
    py_content = f.read()

# Split into logical sections for notebook cells
sections = [
    ("Import Libraries", "# Core Libraries\nimport numpy as np", 35),
    ("ECGDataLoader", "class ECGDataLoader:", 150),
    ("GridAndNoiseFilter", "class GridAndNoiseFilter:", 200),
    ("LeadLocalizationAndDeskew", "class LeadLocalizationAndDeskew:", 250),
    ("SignalTraceExtractor", "class SignalTraceExtractor:", 300),
    ("CalibrationAndScaling", "class CalibrationAndScaling:", 200),
    ("TimeSeriesResampler", "class TimeSeriesResampler:", 150),
    ("Model Architecture", "class ECGSegmentationHead", 250),
    ("Loss Functions", "class DiceLoss", 150),
    ("Training", "class ECGTrainer:", 250),
    ("SNR Metric", "class SNRMetric:", 300),
    ("Inference Pipeline", "class ECGInferencePipeline:", 250),
    ("Robustness Testing", "class ArtifactRobustnessTest:", 300),
]

# Create notebook cells
cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# ECG Image to Time-Series Deep Learning Pipeline\n',
            '## High-Performance 12-Lead ECG Signal Extraction with SNR Optimization\n',
            '\n',
            '**Objective**: Extract 12-lead ECG time-series from images using deep learning.\n',
            '### Key Features:\n',
            '- Multi-task deep learning with artifact correction\n',
            '- Automated lead detection and deskewing\n',
            '- Precise calibration and physical unit conversion\n',
            '- Optimal resampling to fixed sampling rate (500 Hz)\n',
            '- SNR-optimized evaluation with cross-correlation\n',
            '- Robustness testing for various artifacts'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 1. Import Libraries and Setup']
    },
    {
        'cell_type': 'code',
        'metadata': {},
        'source': [
            '# Core Libraries\n',
            'import numpy as np\n',
            'import pandas as pd\n',
            'from pathlib import Path\n',
            'from typing import Tuple, List, Dict, Optional\n',
            'import warnings\n',
            'warnings.filterwarnings("ignore")\n',
            '\n',
            '# Computer Vision\n',
            'import cv2\n',
            'from scipy import ndimage, interpolate, signal\n',
            'from scipy.interpolate import CubicSpline, interp1d\n',
            'from skimage import filters, morphology, measure, transform\n',
            'from skimage.exposure import equalize_adapthist\n',
            '\n',
            '# Deep Learning\n',
            'import torch\n',
            'import torch.nn as nn\n',
            'import torch.optim as optim\n',
            'from torch.utils.data import Dataset, DataLoader\n',
            'import torchvision.models as models\n',
            'import torchvision.transforms as transforms\n',
            '\n',
            '# Utilities\n',
            'import matplotlib.pyplot as plt\n',
            'from tqdm import tqdm\n',
            '\n',
            'np.random.seed(42)\n',
            'torch.manual_seed(42)\n',
            'print("✓ All libraries imported!")'
        ],
        'outputs': [],
        'execution_count': None
    }
]

# Extract and add main classes from Python file
class_starts = {
    'ECGDataLoader': py_content.find('class ECGDataLoader:'),
    'GridAndNoiseFilter': py_content.find('class GridAndNoiseFilter:'),
    'LeadLocalizationAndDeskew': py_content.find('class LeadLocalizationAndDeskew:'),
    'SignalTraceExtractor': py_content.find('class SignalTraceExtractor:'),
    'CalibrationAndScaling': py_content.find('class CalibrationAndScaling:'),
    'TimeSeriesResampler': py_content.find('class TimeSeriesResampler:'),
    'ECGSegmentationHead': py_content.find('class ECGSegmentationHead:'),
    'MultiTaskECGModel': py_content.find('class MultiTaskECGModel:'),
    'DiceLoss': py_content.find('class DiceLoss:'),
    'MultiTaskECGLoss': py_content.find('class MultiTaskECGLoss:'),
    'ECGTrainer': py_content.find('class ECGTrainer:'),
    'SNRMetric': py_content.find('class SNRMetric:'),
    'ECGInferencePipeline': py_content.find('class ECGInferencePipeline:'),
    'ArtifactRobustnessTest': py_content.find('class ArtifactRobustnessTest:'),
}

# Sort by position
sorted_classes = sorted([(name, pos) for name, pos in class_starts.items() if pos != -1], 
                       key=lambda x: x[1])

# Extract each class
for i, (class_name, start_pos) in enumerate(sorted_classes):
    # Find next class or end of file
    if i < len(sorted_classes) - 1:
        end_pos = sorted_classes[i + 1][1]
    else:
        end_pos = len(py_content)
    
    class_code = py_content[start_pos:end_pos].rstrip()
    
    # Create markdown header
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [f'## {class_name} Class']
    })
    
    # Create code cell
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'source': [class_code + '\n\nprint(f"✓ {class_name} defined")'],
        'outputs': [],
        'execution_count': None
    })

# Add demo section
cells.extend([
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## Complete Usage Example and Demo\n\nConfiguration and initialization of the pipeline']
    },
    {
        'cell_type': 'code',
        'metadata': {},
        'source': [
            '# Device configuration\n',
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
            'print(f"Device: {device}")\n',
            '\n',
            '# Create model\n',
            'model = MultiTaskECGModel(\n',
            '    backbone="resnet34",\n',
            '    num_leads=12,\n',
            '    sequence_length=1024,\n',
            '    pretrained=True\n',
            ').to(device)\n',
            '\n',
            'total_params = sum(p.numel() for p in model.parameters())\n',
            'print(f"✓ Model created with {total_params:,} parameters")\n',
            '\n',
            '# Create trainer\n',
            'trainer = ECGTrainer(\n',
            '    model=model,\n',
            '    device=device,\n',
            '    learning_rate=1e-4\n',
            ')\n',
            'print("✓ Trainer initialized")\n',
            '\n',
            '# Create inference pipeline\n',
            'inference_pipeline = ECGInferencePipeline(\n',
            '    model=model,\n',
            '    device=device,\n',
            '    sampling_rate=500.0\n',
            ')\n',
            'print("✓ Inference pipeline ready")\n',
            '\n',
            '# Create robustness tester\n',
            'robustness_tester = ArtifactRobustnessTest(pipeline=inference_pipeline)\n',
            'print("✓ Robustness tester initialized")\n',
            '\n',
            'print("\\n" + "="*60)\n',
            'print("ECG PIPELINE FULLY FUNCTIONAL AND READY")\n',
            'print("="*60)'
        ],
        'outputs': [],
        'execution_count': None
    }
])

# Create notebook
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

# Write notebook
with open('ECG_Pipeline_NEW.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'✓ Notebook created with {len(cells)} cells')
print(f'✓ Classes included: {len([c for c in cells if "Class" in str(c.get("source", ""))])} total')
