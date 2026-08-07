from __future__ import annotations

import argparse

from practical_demo.deployment import DatasetV4DeploymentPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Practical Dataset V4 image-text relation checker."
        )
    )

    parser.add_argument(
        "image",
        help="Path to an automotive-part image.",
    )

    parser.add_argument(
        "description",
        help="Short text description of the automotive part.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = DatasetV4DeploymentPredictor()

    result = predictor.predict_path(
        args.image,
        args.description,
    )

    print()
    print("=== AUTOMOTIVE PART IMAGE-TEXT CHECK ===")
    print("Prediction:", result["label"])
    print(
        "Model score:",
        f"{100.0 * result['model_score']:.2f}%",
    )

    print()
    print("Class scores:")

    for label, score in result["scores"].items():
        print(
            f"  {label:<14}"
            f"{100.0 * score:6.2f}%"
        )

    print()
    print(
        "Recognized text features:",
        result["recognized_text_features"],
    )

    if result["warning"]:
        print("Warning:", result["warning"])

    print()
    print(result["note"])


if __name__ == "__main__":
    main()
