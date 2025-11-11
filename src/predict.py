import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run object detection predictions")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path to image/video or directory"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for inference"
    )
    parser.add_argument(
        "--output", type=str, default="predictions", help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    model = YOLO(args.weights)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(output_dir),
    )

    print(f"Predictions completed! Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
