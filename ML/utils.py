import os
from datetime import datetime

from pytz import timezone


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)

    return file_paths


def get_missing():
    return [
        26795,
        26799,
        26800,
        26807,
        26813,
        26823,
        26827,
        26851,
        26854,
        26857,
        26858,
        26970,
        26979,
        26998,
        27002,
        27018,
        27030,
        27037,
        27090,
        27134,
        27168,
        27179,
        27219,
        27252,
        27270,
        27324,
        27403,
        27429,
        27478,
        27522,
        27525,
        27589,
        27614,
        27627,
        27648,
        27649,
        27662,
        27667,
        27713,
        27778,
        27851,
        27908,
        27919,
        27924,
        27940,
        28041,
        28062,
        28137,
        28156,
        28179,
        28254,
        28256,
        28362,
        28527,
        28543,
        28559,
        71505,
        71545,
        71592,
        71610,
        71621,
        71626,
        71642,
        71771,
        71778,
        71801,
        71831,
        71834,
        71837,
        71850,
        71883,
        71890,
        71894,
        71898,
        71925,
        72010,
        72012,
        72029,
        72110,
        72120,
        72123,
        72138,
        72153,
        72191,
        72222,
        72263,
        72317,
        72323,
        72349,
        72459,
        72537,
        72603,
        72615,
        81818,
    ]
