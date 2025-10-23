#!/usr/bin/env python3
"""
Test script to verify ECG Pipeline components work correctly
"""

import sys
import traceback

def test_imports():
    """Test that all required imports work"""
    try:
        import numpy as np
        import pandas as pd
        import cv2
        from scipy import ndimage, interpolate, signal
        from scipy.interpolate import CubicSpline, interp1d
        from skimage import filters, morphology, measure, transform
        from skimage.exposure import equalize_adapthist
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import torchvision.models as models
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Rectangle
        import seaborn as sns
        from tqdm import tqdm
        import json
        import pickle

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_classes():
    """Test that main classes can be instantiated"""
    try:
        import torch
        from ECG_Pipeline import (
            ECGDataLoader, GridAndNoiseFilter, LeadLocalizationAndDeskew,
            SignalTraceExtractor, CalibrationAndScaling, TimeSeriesResampler,
            MultiTaskECGModel, DiceLoss, JaccardLoss, MultiTaskECGLoss,
            ECGTrainer, SNRMetric, ECGInferencePipeline, ArtifactRobustnessTest
        )

        # Test basic instantiation
        data_loader = ECGDataLoader(image_dir='', csv_dir='')
        grid_filter = GridAndNoiseFilter()
        lead_localizer = LeadLocalizationAndDeskew()
        trace_extractor = SignalTraceExtractor()
        calibrator = CalibrationAndScaling()
        resampler = TimeSeriesResampler()

        # Test model creation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiTaskECGModel(backbone='resnet34', num_leads=12, sequence_length=1024)

        # Test loss functions
        dice_loss = DiceLoss()
        jaccard_loss = JaccardLoss()
        multi_loss = MultiTaskECGLoss()

        # Test trainer
        trainer = ECGTrainer(model, device)

        # Test metrics
        snr_metric = SNRMetric()

        # Test pipeline
        pipeline = ECGInferencePipeline(model, device)

        # Test robustness
        robustness = ArtifactRobustnessTest(pipeline)

        print("✓ All classes instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Class instantiation error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    try:
        import numpy as np
        import torch
        from ECG_Pipeline import generate_synthetic_ecg_image, MultiTaskECGModel, SNRMetric

        # Generate test image
        image, metadata = generate_synthetic_ecg_image(height=256, width=512, num_leads=3)
        print(f"✓ Synthetic image generated: shape {image.shape}")

        # Test model forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiTaskECGModel(backbone='resnet34', num_leads=12, sequence_length=1024).to(device)

        # Create dummy input
        dummy_input = torch.randn(1, 1, 256, 512).to(device)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Model forward pass successful: {list(output.keys())}")

        # Test SNR calculation
        from ECG_Pipeline import SNRMetric
        snr_calc = SNRMetric()

        pred = np.random.randn(1000) * 0.1 + np.sin(np.linspace(0, 4*np.pi, 1000))
        truth = np.sin(np.linspace(0, 4*np.pi, 1000))

        snr = snr_calc.calculate_snr(pred, truth)
        print(f"✓ SNR calculation: {snr:.2f} dB")

        return True
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("ECG PIPELINE - COMPREHENSIVE TEST SUITE")
    print("="*60)

    tests = [
        ("Import Test", test_imports),
        ("Class Instantiation Test", test_classes),
        ("Basic Functionality Test", test_basic_functionality)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            print("✓ PASSED")
        else:
            print("✗ FAILED")

    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("🎉 ALL TESTS PASSED - Pipeline is ready for use!")
        return 0
    else:
        print("❌ Some tests failed - check errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())