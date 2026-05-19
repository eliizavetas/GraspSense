import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import GraspSensePipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GraspSense pipeline."
    )

    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help='Natural language command, e.g. "Take a paper cup carefully".',
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the RGB input image.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Directory for generated outputs.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"[GraspSense] Input image does not exist: {image_path}")
        print("[GraspSense] Put an RGB image into data/input/ or pass a valid --image path.")
        return

    pipeline = GraspSensePipeline()
    result = pipeline.run(
        command=args.command,
        image_path=str(image_path),
        output_dir=args.output_dir,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
