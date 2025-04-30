# Chest X-Ray Classification System

A deep learning system for classifying chest X-ray images using TensorFlow. The system provides both training capabilities and a FastAPI-based web interface for inference.

## Features

### 1. Model Training
- DenseNet-121 based architecture
- Automatic validation split (15% if not provided)
- Comprehensive training metrics (accuracy, AUC)
- Early stopping and learning rate reduction
- Mixed precision training support
- Multiple backbone options (DenseNet121, DenseNet169, EfficientNetB3)

### 2. Web Interface
- FastAPI-based web server
- Real-time image classification
- Confidence scoring
- User-friendly interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd xray_detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.10 or higher
- TensorFlow 2.16.2 or higher
- Required packages are listed in `requirements.txt`

## Usage

### Training the Model
```bash
./train.sh
```
This will:
- Train the model on your dataset
- Save the best model as `chest_xray_best.keras`
- Generate training logs in the `runs` directory

### Running the Web Interface
```bash
./run.sh
```
This will:
- Start the FastAPI server
- Access the interface at `http://localhost:8501`

## Project Structure

```
xray_detection/
├── app.py                 # FastAPI application
├── train_eval_tf.py       # Training and evaluation script
├── requirements.txt       # Package dependencies
├── train.sh              # Training script
├── run.sh                # Server startup script
├── README.md             # Project documentation
├── static/               # Static web assets
├── templates/            # Web templates
└── runs/                 # Model checkpoints and logs
```

## Model Information

The system uses a DenseNet-121 based deep learning model with the following features:
- Pre-trained on ImageNet
- Fine-tuned for chest X-ray classification
- Binary classification (Normal vs Pneumonia)
- Regularization: augmentation + dropout + early-stopping
- Mixed precision training support

## Development

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Your chosen license]