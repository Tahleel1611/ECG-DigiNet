#!/usr/bin/env python3
"""
Simple usage example for the ECG Pipeline
Demonstrates basic usage patterns for the ECG image-to-time-series pipeline
"""

import torch
import numpy as np
from ECG_Pipeline import (
    generate_synthetic_ecg_image,
    MultiTaskECGModel,
    ECGInferencePipeline,
    SNRMetric
)

def main():
    print("🫀 ECG Pipeline - Usage Example")
    print("=" * 40)

    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Generate synthetic ECG data for testing
    print("\n📊 Generating synthetic ECG data...")
    image, metadata = generate_synthetic_ecg_image(
        height=512, width=1024, num_leads=12, add_grid=True, add_noise=True
    )
    print(f"✓ Generated ECG image: {image.shape}")

    # 3. Create and load the model
    print("\n🤖 Creating multi-task model...")
    model = MultiTaskECGModel(
        backbone='resnet34',
        num_leads=12,
        sequence_length=1024,
        pretrained=True
    ).to(device)
    print("✓ Model created successfully")

    # 4. Run inference
    print("\n🔍 Running model inference...")
    with torch.no_grad():
        # Prepare input
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

        # Forward pass
        outputs = model(input_tensor)

        print("✓ Inference completed:")
        print(f"  - Segmentation mask: {outputs['segmentation'].shape}")
        print(f"  - Bounding boxes: {outputs['bboxes'].shape}")
        print(f"  - Confidences: {outputs['confidences'].shape}")
        print(f"  - Signal traces: {outputs['signal_traces'].shape}")

    # 5. Create inference pipeline
    print("\n🔬 Setting up inference pipeline...")
    pipeline = ECGInferencePipeline(model, device, sampling_rate=500.0)
    print("✓ Pipeline ready for use")

    # 6. Demonstrate SNR calculation
    print("\n📈 Testing SNR calculation...")
    snr_calculator = SNRMetric(sampling_rate=500.0, max_time_shift=0.2)

    # Create synthetic ground truth and prediction
    np.random.seed(42)
    pred_signal = np.sin(np.linspace(0, 4*np.pi, 1000)) + 0.1 * np.random.randn(1000)
    true_signal = np.sin(np.linspace(0, 4*np.pi, 1000))

    snr_value = snr_calculator.calculate_snr(pred_signal, true_signal)
    print(f"✓ SNR calculation: {snr_value:.2f} dB")
    # 7. Show how to use with real data
    print("\n💡 Usage with real ECG images:")
    print("""
    # Load your ECG image
    import cv2
    image = cv2.imread('your_ecg_image.jpg', cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Process through pipeline
    result = pipeline.process_single_image('your_ecg_image.jpg')

    # Extract signals
    signals = result['resampled_signals']
    print(f"Extracted {len(signals)} leads")

    # Evaluate quality
    snr_results = pipeline.evaluate_with_ground_truth(
        ['your_ecg_image.jpg'],
        ['ground_truth.csv']
    )
    print(f"Mean SNR: {snr_results['mean_snr_overall']:.2f} dB")
    """)

    print("\n" + "=" * 40)
    print("🎉 Pipeline is ready for production use!")
    print("Use the classes above to process your ECG images.")
    print("=" * 40)

if __name__ == "__main__":
    main()