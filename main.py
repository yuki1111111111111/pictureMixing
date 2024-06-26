import configparser
from predicted import predicted
from expand_mask import expand_mask
from MaskColorExtract import MaskColorExtract
from stitch import stitch


def run_pipeline(step):
    config_file = "mixConfig.ini"
    config = configparser.ConfigParser()
    config.read(config_file)

    if step == "predicted":
        predicted()

    elif step == "expand":
        expand_mask()

    elif step == "extract":
        MaskColorExtract()

    elif step == "stitch":
        stitch()

    elif step == "all":
        print("predicting...")
        predicted()
        print("mask expanding...")
        expand_mask()
        print("extracting color pictures...")
        MaskColorExtract()
        print("picture stitching...")
        stitch()
