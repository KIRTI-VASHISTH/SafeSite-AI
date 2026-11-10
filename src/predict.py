
import argparse
from pathlib import Path

from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "safesite_yolov8n_cpu"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "runs" / "predict"


# --------------------------------------------------
# 2. PREDICTION FUNCTION
# --------------------------------------------------

def run_prediction(source, confidence, show, run_name):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # Load trained YOLO model
    model = YOLO(str(MODEL_PATH))

    print("Loaded model:", MODEL_PATH)
    print("Input source:", source)
    print("Confidence threshold:", confidence)

    # Run object detection
    results = model.predict(
        source=source,
        conf=confidence,
        imgsz=640,
        device="cpu",
        save=True,
        show=show,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True
    )

    print("\nPrediction completed.")
    print("Output directory:", OUTPUT_DIR / run_name)

    return results


# --------------------------------------------------
# 3. COMMAND-LINE ARGUMENTS
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SafeSite PPE Detection"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, folder, video path, or webcam index (0)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display predictions in a window"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Name of the prediction output folder"
    )

    args = parser.parse_args()

    # Convert webcam index from string to integer
    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    # Validate non-webcam input
    if not isinstance(source, int):
        if not Path(source).exists():
            raise FileNotFoundError(
                f"Input source not found: {source}"
            )

    if not 0 < args.conf < 1:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    run_prediction(
        source=source,
        confidence=args.conf,
        show=args.show,
        run_name=args.name
    )

