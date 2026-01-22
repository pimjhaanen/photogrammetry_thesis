# LEI Kite Photogrammetry and UWB Ranging System

This repository contains the code for determining the shape of a **Leading Edge Inflatable (LEI)** kite in flight using **stereoscopic photogrammetry** and **ultra-wideband (UWB) ranging**. These methods are time-synchronized and combined with **KCU (Kite Control Unit)** and **Pitot tube data** to provide accurate 3D reconstructions of the kite's deformation during flight. The system is designed to analyze the kite's behavior and aerodynamic characteristics by combining several subsystems that include sensor data processing, image analysis, and synchronization.

The code was used for the graduation project of Pim Julius Haanen, which is found on https://resolver.tudelft.nl/uuid:40f2626a-a436-4068-8269-3c663fc249fd. The report gives a detailed explanation on the steps taken and why some pieces of code were necessary. The code uses the flight data available on https://github.com/awegroup/Flightdata09102025 and this is also what the results are based on. 

The code is divided into several key subsystems:

---

## Subsystems Overview

### 1. KCU Pitot Subsystem

The **KCU Pitot Subsystem** processes **wind tunnel experiment data**, focusing on the analysis and calibration of **Pitot tubes**, **wind speed**, and **angle of attack (AoA)** measurements. It plays a crucial role in ensuring that the aerodynamic parameters are measured accurately for the kite's performance analysis.

#### Main Functions:
1. **Pitot Calibration and Data Processing**  
   - **File**: `pitot_noise_test.py, calibration_pitot.py`  
   - **Purpose**: Filters noise from sensor data collected during wind tunnel experiments and calibrates **Pitot measurements** using a **linear model**.

2. **New Vane Testing**  
   - **File**: `new_vane_testing.py`  
   - **Purpose**: Analyzes the performance of redesigned **angular vanes**. Tests their responsiveness to varying flow conditions and assesses their integration with the **Pitot system**.

3. **Main Data Processing**  
   - **File**: `main_KCU_pitot.py`  
   - **Purpose**: Combines results from the **Pitot**, **angular vanes**, and **KCU system**. Applies **calibration**, **data filtering**, and generates a cleaned dataset ready for further analysis.

4. **Visualizations and Results Analysis**  
   - **Purpose**: Generates visualizations to compare the raw and calibrated data, ensuring the results are aligned with expected physical behaviors.

---

### 2. Photogrammetry Subsystem

The **Photogrammetry Subsystem** is responsible for accurately measuring the 3D shape of the kite by detecting and analyzing markers placed on it. It integrates various subcomponents, including **accuracy analysis**, **marker detection**, **calibration**, and **synchronization**.

#### Main Functions:

1. **Accuracy Analysis**  
   - **Files**: `accuracy_test_gridwise.py`, `gridwise_plotting_functions.py`  
   - **Purpose**: Compares detected marker positions with an ideal reference grid. This subsystem calculates **Euclidean errors** and visualizes discrepancies to ensure precision in marker placement.

2. **Calibration**  
   - **Files**: `calibrate_charuco.py`, `calibrate_checkerboard.py`, `calibrate_circles.py`, `calibrate_fisheye.py`  
   - **Purpose**: Performs **intrinsic** (single camera) and **extrinsic** (stereo camera) calibration to adjust for lens distortion and stereo alignment.
     - **Intrinsic Calibration**:
       - **Charuco boards**: `calibrate_charuco.py`
       - **Checkerboards**: `calibrate_checkerboard.py`
       - **Asymmetric circles**: `calibrate_circles.py`
     - **Extrinsic Calibration**:
       - **Stereo calibration**: `stereoscopic_calibration.py`
       - **Stereo calibration quality check**: `stereo_calibration_quality.py`
       - **Fisheye lens calibration**: `stereo_calibration_fisheye.py`

   - **Output**: Saves calibrated values in `.pkl` files, with undistorted images provided for verification.

3. **Marker Detection**  
   - **Files**: `detect_crosses.py`, `detection_test_without_crosses.py`, `centre_detection.py`, `check_HSV_colour.py`, `marker_detection_utils.py`  
   - **Purpose**: Detects markers (e.g., red crosses, circular blobs) in stereo images, refining their positions with **subpixel accuracy** to ensure reliable 3D reconstruction.

## 4. **Static Experiments**  

   **Files**:  
   `correction_factor_calculator.py`, `span_calculator.py`, `static_processing.py`, `static_processing_fisheye.py`, `anhedral_angle_calculator.py`, `determine_pitch_angle.py`, `determine_zeta_xi.py`, `twist_and_billowing.py`, `plot_static_results.py`  

   **Purpose**:  
   The **Static Experiments** subsystem processes **static frames** by manually clicking matching markers in both the left and right frames of stereo pairs. The objective is to generate **3D point clouds** for further analysis, focusing on a variety of critical angles and deformations in the kite. This subsystem is designed to work with **non-moving (static) frames**, and the following steps are carried out for processing:

   - **Correction Factor Calculation**:  
     - **File**: `correction_factor_calculator.py`  
     - **Purpose**: This script computes the **correction factors** required to compensate for **bar bending** and **twist** in the kite's structure. It adjusts the yaw, roll, and pitch angles of the cameras by minimizing **vertical disparity** and **span deviation** from UWB data, thus improving the accuracy of 3D point cloud generation.

   - **Span Check**:  
     - **File**: `span_calculator.py`  
     - **Purpose**: This script offers a quick check to validate the **span measurement** of the kite. It computes the **pixel coordinates** in the stereo images and compares them against the expected span values, ensuring the accuracy of the calibration.

   - **3D Point Cloud Generation**:  
     - **Files**: `static_processing.py`, `static_processing_fisheye.py`  
     - **Purpose**: These scripts allow users to manually click matching markers in the **left** and **right** frames, generating a **3D point cloud**. The **static_processing.py** is the recommended approach for standard cameras, while **static_processing_fisheye.py** can be used for fisheye lens calibration, although its use is not recommended unless specifically required.

   - **Calculation of Critical Angles and Shear**:  
     The following scripts calculate the deformation characteristics of the kite using the generated 3D point clouds:
     - **Anhedral Angle**:  
       - **File**: `anhedral_angle_calculator.py`  
       - **Purpose**: This script computes the **anhedral angle** of the kite’s wing by analyzing the midpoints of the left and right wing tips in the yz-plane. This angle provides insight into the wing’s tilt relative to the horizontal plane.
     - **Pitch Angle**:  
       - **File**: `determine_pitch_angle.py`  
       - **Purpose**: Calculates the **pitch angle** of the kite’s wing by analyzing the average pitch of the wing’s struts. This angle is essential for understanding the orientation of the kite in flight.
     - **Shear Angles**:  
       - **File**: `determine_zeta_xi.py`  
       - **Purpose**: This script calculates the **front shear (zeta)** and **bottom shear (xi)** angles by analyzing changes in the tip-line angles across both the y–x and z–y planes. These angles are used to quantify the deformation behavior of the kite under varying flight conditions.

   - **Deformation Analysis and Plotting**:  
     - **Files**: `twist_and_billowing.py`, `plot_static_results.py`  
     - **Purpose**: The **twist_and_billowing.py** script calculates the **twist** and **billowing** of the kite during the static frame analysis. It generates different output files that provide insight into how the kite deforms. The **plot_static_results.py** script is responsible for visualizing the static test results by plotting the **3D reconstructed shapes** and deformation characteristics of the kite, enabling a thorough analysis of the static experiment.

## 5. **Photogrammetry Main Functions**  

   **Files**:  
   `determine_baselength.py`, `kite_shape_reconstruction_utils.py`, `main_photogrammetry.py`, `stereo_photogrammetry_utils.py`  

   **Purpose**:  
   The **Photogrammetry Subsystem** processes stereo images to generate accurate **3D point clouds** of the kite, which are crucial for understanding its deformation and geometry in flight. This subsystem is responsible for the key photogrammetry operations, including **camera calibration**, **feature matching**, and **3D reconstruction**. The following are the key functions:

   - **Baseline Length Calculation**:  
     - **File**: `determine_baselength.py`  
     - **Purpose**: This script computes the **baseline length** required for accurate stereo vision measurements. It calculates the **depth error** as a function of the baseline distance and analyzes the effect of baseline length on depth precision. This is essential to determine the optimal stereo camera setup for minimizing depth errors during 3D reconstruction.

   - **Kite Shape Reconstruction**:  
     - **File**: `kite_shape_reconstruction_utils.py`  
     - **Purpose**: This script contains utilities for reconstructing the **kite’s shape** from the marker detections in stereo images. It processes the detected markers and fits the **leading edge (LE)** and **struts** into 3D space. It also provides essential utilities for triangulating points, estimating angles, and fitting 3D lines and planes to the kite's structure, enabling accurate modeling of the kite’s deformation.

   - **Main Photogrammetry Pipeline**:  
     - **File**: `main_photogrammetry.py`  
     - **Purpose**: This is the central script for processing stereo frames in the photogrammetry pipeline. It reads camera calibration data, processes stereo image pairs, applies **rotation corrections**, performs **feature matching**, and triangulates points to generate the 3D structure of the kite. It ties together all processes from calibration to triangulation, generating the final 3D point cloud used for analysis.

   - **Stereo Photogrammetry Utilities**:  
     - **File**: `stereo_photogrammetry_utils.py`  
     - **Purpose**: This script provides various utilities for stereo image processing, such as **feature tracking**, **rectification**, and **triangulation**. It handles stereo pair processing with **rotation corrections** and includes options for visualizing stereo frames with overlaid markers and diagnostic statistics, such as **epipolar line alignment**. These utilities are essential for ensuring that stereo image pairs are properly aligned, enabling precise 3D reconstruction of the kite’s geometry.

---

### 3. Synchronization Subsystem (integrated in photogrammetry folder)

The **Synchronization Subsystem** ensures that data from different video and sensor sources (e.g., UWB, cameras) are aligned in time, allowing for accurate correlation and 3D reconstruction.

#### Main Functions:

1. **Video Merging**  
   - **File**: `merge_multiple_videos.py`  
   - **Purpose**: Merges **split video files** from GoPro cameras into one continuous stream.

2. **Drift Testing**  
   - **File**: `gopro_drift_test.py`  
   - **Purpose**: Measures **drift** between two GoPro videos by comparing their **audio tracks**, adjusting for any time offsets.

3. **Synchronization Utilities**  
   - **File**: `synchronisation_utils.py`  
   - **Purpose**: Synchronizes **photogrammetry cameras** with **UWB data** using timestamp adjustments and calibration.

4. **Audio Plotting**  
   - **File**: `audio_plotting_utils.py`  
   - **Purpose**: Visualizes and analyzes **audio data** from videos, ensuring synchronization through waveforms and spectrograms.

---

### 4. UWB Subsystem

The **UWB Subsystem** processes distance measurements from **UWB sensors**, applying calibration, filtering, and noise reduction techniques to ensure accurate positional data.

#### Main Functions:

1. **Calibration**  
   - **Files**: `calibration_UWB.py`, `ranging_accuracy_uwb.py`, `static_measurement.py`, `range_test.py`, `fresnel_zone_calculation.py`  
   - **Purpose**:  
     - **`calibration_UWB.py`**: Performs **linear calibration** of the UWB sensors to correct raw distance measurements based on a reference.
     - **`ranging_accuracy_uwb.py`**: Analyzes **sensor accuracy** by determining the consistency of UWB readings over time.
     - **`static_measurement.py`**: Takes **static UWB measurements** (5-second averages) for calibration under stationary conditions.
     - **`range_test.py`**: Measures the **operational range** of the UWB sensors by assessing distance over time.
     - **`fresnel_zone_calculation.py`**: Calculates the **Fresnel zone radius**, indicating the minimum ground clearance to avoid interference with UWB signals.

2. **Data Processing**  
   - **Files**: `lag_light_wrt_first_log.py`, `noise_visualization_uwb.py`, `postprocess_RAW_UWB_file.py`  
   - **Purpose**:  
     - **`lag_light_wrt_first_log.py`**: Measures the **lag** between UWB signals and external synchronization sources like LED lights.
     - **`noise_visualization_uwb.py`**: **Visualizes UWB noise**, showing variations in the sensor data and filtering out high-frequency noise.
     - **`postprocess_RAW_UWB_file.py`**: Processes **raw UWB data**, applying calibration, interpolation, and smoothing techniques to prepare data for further analysis.

3. **Main Ranging Process**  
   - **File**: `main_UWB.py`  
   - **Purpose**: Starts **UWB ranging**, applying calibration to the data, performing necessary adjustments, and automatically saving the processed measurements for later use in photogrammetry or other subsystems.


---

### 5. Main Post-Processing Pipeline

This pipeline integrates the results from the **KCU + Pitot**, **Photogrammetry**, and **UWB** subsystems and generates a synchronized, processed output. It involves combining the data into a single dataset and visualizing it as a video with relevant metrics overlaid.

#### Main Functions:

1. **combine_all_results.py**  
   - **Function**: Combines the results from the **KCU + Pitot**, **Photogrammetry**, and **UWB** subsystems into a unified Pandas DataFrame. It merges the data based on timestamps or other relevant parameters from each subsystem (e.g., camera frames, sensor data).
   - **Purpose**: Consolidates data for further processing and visualization.

2. **video_generator.py**  
   - **Function**: Generates a **video** from the synchronized and combined data. The video includes:
     - **4 views** of the kite: front, side, bottom, and 3D.
     - **Phase of flight** and telemetry data overlaid on the video.
     - **Synchronized data** (e.g., wind speed, tether force) displayed in real-time.
   - **Purpose**: The video provides a **comprehensive visual representation** of the kite's performance during the test.

