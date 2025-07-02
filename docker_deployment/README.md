# FLARE 2025 MedGemma Docker Deployment

This folder contains Docker deployment files for the FLARE 2025 Medical Multimodal VQA Challenge using MedGemma with fine-tuned adapters.

### Run Inference

```bash
# Run inference on your dataset
docker run --gpus all \
    -v $(pwd)/path/to/dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm medgemma-inference:latest
```

## Build Docker from Source

### Prerequisites

- Docker installed with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- At least 24GB disk space for the Docker image
- Internet connection for downloading models

### Build Docker Image

```bash
# Build the Docker image
docker build -f Dockerfile -t medgemma-inference .
```

> Note: Don't forget the `.` at the end. Models will be downloaded at runtime when needed, making the build process faster and more flexible.

### Alternative: Use Build Script

```bash
# Make the build script executable and run it
chmod +x docker_build.sh
./docker_build.sh
```

## Running Inference

### Direct Docker Run

```bash
# Run inference with volume mounts for input and output
docker run --gpus all \
    -v $(pwd)/path/to/dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    -e HF_TOKEN=your-token \
    --rm medgemma-inference:latest
```

## Expected Input/Output Structure

### Input Directory Structure
```
organized_dataset/
├── testing/
│   ├── Microscopy/
│   ├── Retinography/
│   ├── Ultrasound/
│   └── Xray/
└── validation-public/
    ├── Clinical/
    ├── Dermatology/
    ├── Endoscopy/
    ├── Mammography/
    ├── Microscopy/
    ├── Retinography/
    ├── Ultrasound/
    └── Xray/
```

### Output
The inference will generate a `predictions.json` file in the output directory containing the model's answers to all questions.

## Authentication (Required)

### HuggingFace Token Required

**⚠️ Important**: All MedGemma models (including fine-tuned adapters) require access to the gated base model `google/medgemma-4b-it`. You **must** have a HuggingFace token and access approval.

**Steps to get access:**

1. **Get HuggingFace Token**: Go to [HuggingFace Settings](https://huggingface.co/settings/tokens) and create a token
2. **Request Access**: Apply for access to [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
3. **Wait for Approval**: Google will review your request (can take a few days)
4. **Use Token**: Pass it as environment variable

### Usage Examples

```bash
# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# Using fine-tuned MedGemma adapter (default)
docker run --gpus all \
    -e HF_TOKEN="$HF_TOKEN" \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm medgemma-inference:latest

# Using base MedGemma model
docker run --gpus all \
    -e HF_TOKEN="$HF_TOKEN" \
    -e MODEL_NAME="google/medgemma-4b-it" \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm medgemma-inference:latest
```

## Configuration Options

The Docker container supports several environment variables for configuration:

```bash
# Custom configuration examples
docker run --gpus all \
    -e MODEL_NAME="leoyinn/flare25-medgemma" \
    -e MAX_TOKENS="512" \
    -e VERBOSE="true" \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm medgemma-inference:latest
```

## Memory Requirements

- **GPU Memory**: Minimum 24GB VRAM recommended
- **System RAM**: At least 16GB
- **Docker Container**: Limited to 8GB by default (configurable)

## Testing the Installation

Run the test script to verify everything is working:

```bash
# Test the Docker container
docker run --gpus all --rm medgemma-inference:latest python test_installation.py
```

## Troubleshooting

### Permission Issues
If you encounter permission denied errors:
```bash
chmod -R 777 ./organized_dataset ./predictions
```

### GPU Not Detected
Ensure NVIDIA Container Toolkit is installed:
```bash
# Check if GPU is available in Docker
docker run --gpus all --rm nvidia/cuda:11.8-base nvidia-smi
```

### Out of Memory Errors
- Reduce batch size by setting environment variable: `-e BATCH_SIZE="1"`
- Ensure no other GPU processes are running
- Check available GPU memory: `nvidia-smi`

## Saving and Distributing the Image

### Save Docker Image

```bash
# Save the Docker image for distribution
docker save medgemma-inference:latest -o medgemma-flare2025.tar
gzip medgemma-flare2025.tar
```

### Load Docker Image on Another Machine

```bash
# Load the Docker image on target machine
docker load -i medgemma-flare2025.tar.gz
```

## Model Information

This Docker container uses:
- **Base Model**: google/medgemma-4b-it
- **Fine-tuned Adapter**: leoyinn/flare25-medgemma
- **Training Data**: 19 medical datasets across 7 modalities (CT, MRI, X-ray, Ultrasound, Fundus, Pathology, Endoscopy)
- **LoRA Configuration**: r=16, alpha=32

## Files in this Directory

- `Dockerfile`: Main Docker configuration
- `docker_build.sh`: Automated build script
- `docker_inference.sh`: Container entrypoint script
- `docker-compose.yml`: Docker Compose configuration
- `inference.py`: Main inference script
- `requirements.txt`: Python dependencies
- `.dockerignore`: Docker build ignore patterns

## Citation

If you use this Docker container or the FLARE 2025 fine-tuned model, please cite:

```bibtex
@misc{flare2025-qwenvl,
  title={FLARE 2025 QwenVL Fine-tuned Model for Medical Multimodal VQA},
  author={[Authors]},
  year={2025},
  howpublished={\url{https://github.com/medfm-flare/FLARE25-QWen2.5VL}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [FLARE25-QWen2.5VL Issues](https://github.com/medfm-flare/FLARE25-QWen2.5VL/issues)
- Model Hub: [leoyinn/flare25-qwen2.5vl](https://huggingface.co/leoyinn/flare25-qwen2.5vl) 
