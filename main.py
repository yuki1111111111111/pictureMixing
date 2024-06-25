import argparse
import subprocess

from predicted import predicted
from expand_mask import expand_mask
from MaskColorExtract import MaskColorExtract
from stitch import stitch
import configparser


def main():
    parser = argparse.ArgumentParser(description="Image processing pipeline")
    parser.add_argument(
        "--step",
        choices=["predicted", "expand", "extract", "stitch", "all"],
        required=True,
        help="Step to run",
    )

    args = parser.parse_args()

    config_file = "mixConfig.ini"

    config = configparser.ConfigParser()
    config.read(config_file)

    if args.step == "predicted":
        predicted()

    elif args.step == "expand":
        expand_mask()

    elif args.step == "extract":
        MaskColorExtract()

    elif args.step == "stitch":
        stitch()

    elif args.step == "all":
        print("predicting...")
        predicted()
        print("mask expanding...")
        expand_mask()
        print("extracting color pictures...")
        MaskColorExtract()
        print("picture stitching...")
        stitch()


if __name__ == "__main__":
    main()
