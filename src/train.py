
from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = PROJECT_ROOT / "Data" / "construction-ppe" / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


# --------------------------------------------------
# 2. TRAINING CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4
WORKERS = 2
PATIENCE = 20

DEVICE = "cpu"


# --------------------------------------------------
# 3. TRAINING FUNCTION
# --------------------------------------------------

def train_model():

    # Verify dataset configuration exists
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    print("Dataset configuration:", DATA_YAML)
    print("Training device:", DEVICE)

    # Load pretrained YOLOv8 Nano model
    model = YOLO(MODEL_NAME)

    # Start training
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        patience=PATIENCE,
        device=DEVICE,
        optimizer="auto",
        project=str(RUNS_DIR),
        name="safesite_yolov8n_cpu",
        exist_ok=False,
        pretrained=True,
        amp=False,
        cache=False,
        plots=True,
        save=True,
        val=True,
        seed=42,
        verbose=True
    )

    print("\nTraining completed.")
    print("Results directory:", results.save_dir)


# --------------------------------------------------
# 4. SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    train_model()
