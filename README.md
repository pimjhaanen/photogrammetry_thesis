This code is used to to determine the shape of a LEI kite in flight. It uses stereoscopic photogrammetry (through OpenCV library) together with ultra-wideband ranging (UWB) for this purpose. These are time-synchronised, and combined with KCU and Pitot tube data.

The code consists of four subsystems:

**KCU Pitot Subsystem**

The KCU Pitot Subsystem is integral to processing wind tunnel experiment data, focusing on the analysis and calibration of pitot tubes, wind speed, and angle of attack (AoA) measurements. This subsystem is housed within the KCU_pitot folder and includes several components that facilitate the analysis of both initial and ongoing wind tunnel campaigns. The primary tasks handled within this subsystem include:

Pitot Calibration and Data Processing: The pitot_noise_test.py script processes and filters noise from sensor data collected during wind tunnel experiments, especially focusing on wind speed and AoA measurements. The calibration procedure is performed by applying a linear model to correct the measured pitot values using reference data.

New Vane Testing: The new_vane_testing.py script analyzes the performance of redesigned angular vanes. These vanes are tested for their responsiveness to varying flow conditions, and the results are used to assess their suitability for integration with the pitot system.

Main Data Processing: The main_KCU_pitot.py script consolidates results from the pitot, angular vanes, and KCU system. It performs calibration, filtering, and data processing, outputting a cleaned dataset ready for further analysis. This script ensures that the collected data is consistently processed and aligned with the experimental goals.

Visualizations and Results Analysis: The subsystem also generates visualizations to compare the raw and calibrated data, ensuring the results are aligned with expected physical behaviors. The processed data is stored and can be used for validation purposes in future experiments.

