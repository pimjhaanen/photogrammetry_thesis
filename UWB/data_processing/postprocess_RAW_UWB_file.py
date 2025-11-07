"This file can be used to process RAW files that haven't been processed automatically in the main_UWB code"

from UWB.main_UWB import apply_postprocessing
import os

raw_csv_path = "../output/uwb_flight_20251009_153306_raw.csv"
calibration_path = "../calibration/uwb_calibration.json"

if __name__ == "__main__":
    post_path = apply_postprocessing(os.path.abspath(raw_csv_path),
                                     calibration_path=calibration_path,
                                     apply_low_pass=True, alpha=0.95)