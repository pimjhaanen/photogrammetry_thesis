This code is used to to determine the shape of a LEI kite in flight. It uses stereoscopic photogrammetry (through OpenCV library) together with ultra-wideband ranging (UWB) for this purpose. These are time-synchronised, and combined with KCU and Pitot tube data.

The code consists of four subsystems:

**KCU Pitot Subsystem**

The KCU Pitot Subsystem is integral to processing wind tunnel experiment data, focusing on the analysis and calibration of pitot tubes, wind speed, and angle of attack (AoA) measurements. This subsystem is housed within the KCU_pitot folder and includes several components that facilitate the analysis of both initial and ongoing wind tunnel campaigns. The primary tasks handled within this subsystem include:

Pitot Calibration and Data Processing: The pitot_noise_test.py script processes and filters noise from sensor data collected during wind tunnel experiments, especially focusing on wind speed and AoA measurements. The calibration procedure is performed by applying a linear model to correct the measured pitot values using reference data.

New Vane Testing: The new_vane_testing.py script analyzes the performance of redesigned angular vanes. These vanes are tested for their responsiveness to varying flow conditions, and the results are used to assess their suitability for integration with the pitot system.

Main Data Processing: The main_KCU_pitot.py script consolidates results from the pitot, angular vanes, and KCU system. It performs calibration, filtering, and data processing, outputting a cleaned dataset ready for further analysis. This script ensures that the collected data is consistently processed and aligned with the experimental goals.

Visualizations and Results Analysis: The subsystem also generates visualizations to compare the raw and calibrated data, ensuring the results are aligned with expected physical behaviors. The processed data is stored and can be used for validation purposes in future experiments.

**Photogrammetry Subsystem**

The photogrammetry subsystem consists of several subsystems in itself. It hosts an accuracy analysis, calibration, codes for marker detection, the static testing campaing and synchronisation.

**Photogrammetry accuracy**

The accuracy subsystem evaluates the precision of the photogrammetry system by comparing the detected marker positions with ideal reference points arranged in a grid.

Matching Detected Points: The accuracy_test_gridwise.py script matches the measured markers to an ideal reference grid, minimizing the Euclidean distance between corresponding points.

Error Calculation: It calculates the error between the detected and ideal positions of the markers, providing a measure of system accuracy.

Error Visualization: The gridwise_plotting_functions.py script visualizes the discrepancies by plotting the markers in 3D space, showing the alignment errors and helping to identify areas where the system needs improvement.

**Photogrammetry Calibration**

The calibration subsystem in OpenCV performs both intrinsic and extrinsic calibration for single and stereo camera setups.

Intrinsic Calibration (Single Camera)

The calibration process for single cameras involves using different patterns (checkerboards, Charuco boards, or asymmetric circles) to determine the intrinsic parameters of the camera, including the camera matrix and distortion coefficients.

Scripts:

calibrate_charuco.py – Uses Charuco boards to refine calibration with ArUco markers.

calibrate_checkerboard.py – Uses checkerboards to calibrate the camera.

calibrate_circles.py – Uses asymmetric circle grids for calibration.

calibrate_fisheye.py – Specifically for fisheye lens calibration.

Output: Calibrated values are saved as .pkl files and can also output undistorted images.

Extrinsic Calibration (Stereo Camera)

This step involves matching frames from the left and right cameras, which is essential for stereo calibration.

Scripts:

stereoscopic_calibration.py – Performs stereo calibration by calculating intrinsic parameters for both cameras and their relative pose (rotation and translation).

stereo_calibration_quality.py – Evaluates the quality of the stereo calibration by checking reprojection errors and epipolar distances.

stereo_calibration_fisheye.py – Calibration for fisheye lenses in stereo setups, though not recommended for general use.

Data Export and Visualization

Scripts:

export_stereo_staes.py – Exports calibration results for visualization, such as raw, undistorted, rectified, and epipolar-aligned images.

**Photogrammetry marker detection**

detect_crosses.py

Function: Detects red cross markers in video frames. It uses HSV segmentation to isolate red regions, then applies morphological operations (e.g., opening, closing) to clean the mask. It then finds contours of the mask and refines the marker center using subpixel accuracy.

Purpose: Determines if and where red cross markers appear in video frames, essential for tracking marker positions in 3D reconstruction.

detection_test_without_crosses.py

Function: A variant of the marker detection test that checks for general marker detection without relying on red crosses. It analyzes video frames to detect red color segments and tests various marker sizes to determine the minimum size in pixels for detection.

Purpose: Helps determine the minimum marker size required for detection, ensuring the system can accurately identify markers in different conditions.

centre_detection.py

Function: This code allows users to interactively pick pixel locations on frames and analyze the HSV values of those pixels. It performs brightness/contrast boosting and scaling for better visualization, and allows the user to click on regions of interest in a frame to obtain the HSV values.

Purpose: Facilitates the calibration of the marker detection process by helping determine accurate color ranges for different markers.

check_HSV_colour.py

Function: Interactively checks the HSV values of specific regions in video frames. Users can click on areas of the image, and the script will report the corresponding HSV values.

Purpose: This helps refine color thresholds (e.g., for red markers) by checking pixel values in the context of the frame, ensuring that the right marker color is targeted.

marker_detection_utils.py

Function: This utility script contains functions for detecting circular blobs (e.g., for markers), refining their subpixel centers, and estimating ArUco marker poses. It includes blob detection using HSV masking and subpixel refinement using Gaussian fitting.

Purpose: Supports high-precision marker detection, crucial for accurate marker positioning and 3D reconstruction in photogrammetry workflows.

**Photogrammetry static experiments**

This subsystem focuses on processing static frames manually by clicking matching markers in the left and right images of a stereo pair, generating a 3D point cloud from these clicks. Here’s what each script does:

correction_factor_calculator.py

Function: Computes the correction factors needed for bar bending and twist. It adjusts the yaw, roll, and pitch angles of the cameras by minimizing the vertical disparity and span deviation from UWB (Ultra-Wideband) data, ensuring accurate 3D reconstruction from stereo frames.

span_calculator.py

Function: Provides a quick check for the span measurement. It computes the pixel coordinates from the stereo images and compares them against expected span values, helping verify the results of the stereo calibration and ensuring consistency.

static_processing.py

Function: Lets you manually click matching markers in left and right images. The script then uses these clicks to generate a 3D point cloud, which is saved to the static_test_output folder. These point clouds are used for analysis and are part of the results discussed in the report.

Output: 3D point clouds in CSV format for further analysis.

static_processing_fisheye.py

Function: A variant of static_processing.py designed to handle fisheye lens calibration. It's only recommended when the camera system is calibrated with fisheye lenses.

anhedral_angle_calculator.py

Function: Computes the anhedral angle of the wing by analyzing the midpoints of the left and right wing tips in the yz-plane. This angle provides insights into the wing’s tilt relative to the horizontal plane.

determine_pitch_angle.py

Function: Calculates the pitch angle of the wing by analyzing the average pitch of the wing struts, providing a measure of the wing’s overall orientation.

determine_zeta_xi.py

Function: Computes the front shear (zeta) and bottom shear (xi) angles based on the tip-line angle changes in both the y–x and z–y planes. These angles help in understanding the deformation behavior of the wing under different conditions.

twist_and_billowing.py

Function: Calculates the twist and billowing of the wing based on the markers' positional data. This is crucial for understanding the aerodynamic characteristics of the kite and how it deforms during operation.

plot_static_results.py

Function: Provides a visualization of the static results, plotting the reconstructed 3D shapes and deformation characteristics of the wing. It helps in analyzing the quality and accuracy of the static frame processing.

**Photogrammetry synchronisation**

The synchronization subsystem ensures that different video and sensor data sources are aligned in time, making it possible to correlate measurements and generate accurate 3D reconstructions. The subsystem includes several components:

merge_multiple_videos.py

Function: Merges videos automatically split by GoPro cameras (e.g., GX01, GX02, etc.). This script finds and concatenates multiple video files from a GoPro camera into a single file.

Use: Helps process videos that are split across several files, ensuring they are combined into one continuous video for easier analysis.

gopro_drift_test.py

Function: Detects and measures drift between two GoPro videos by extracting and comparing the audio tracks. It uses cross-correlation to compute the time lag (drift) between the two audio streams, aligning the videos based on their audio tracks.

Use: Ensures the synchronization of two GoPro videos by calculating the drift and adjusting for any time offset between the videos.

synchronisation_utils.py

Function: This script synchronizes photogrammetry cameras with one another and with UWB data. It uses calibration and timestamp adjustments to align the data sources accurately. It is a crucial part of the main processing pipeline.

Use: Used to handle and correct timing mismatches between different cameras and sensors (e.g., UWB), ensuring accurate data fusion for further analysis.

audio_plotting_utils.py

Function: This code extracts mono audio from video files, applies a band-pass filter to isolate specific frequencies, and plots the filtered audio waveform and its spectrogram.

Use: Helpful for visualizing and analyzing the audio data from the videos, ensuring synchronization through visual inspection of the audio waveforms and frequency content.

**Others**

Lastly there are some separate codes in the photogrammetry pipleine. determine_baselength.py, kite_shape_reconstruction_utils.py, main_photogrammetry.py and stereo_photogrammetry_utils.py, which are all the main codes of photogrammetry.

1. determine_baselength.py

Function: This script computes the baseline length required for stereo vision accuracy. It calculates depth error as a function of the baseline distance and analyzes the effect of baseline length on depth precision.

Purpose: It helps determine the optimal baseline length for stereo cameras to minimize depth errors, ensuring accurate 3D reconstruction from stereo images.

2. kite_shape_reconstruction_utils.py

Function: This script contains utilities for reconstructing the kite’s shape from marker detections in stereo images. It processes the detected markers, identifies the leading edge (LE) and struts, and fits the geometry of the kite in 3D space.

Key Functions:

separate_LE_and_struts: Labels detected markers as either part of the leading edge (LE) or as strut identifiers (0-7), based on brightness thresholds and horizontal spacing. It clusters markers and assigns labels accordingly.

3D utilities: It provides essential utilities for triangulating points, estimating angles, and fitting 3D lines and planes to the kite's structure.

Purpose: This utility is vital for converting 2D image data into 3D shapes, enabling accurate modeling of the kite's deformation and geometry.

3. main_photogrammetry.py

Function: This is the central script for processing stereo frames in the photogrammetry pipeline. It loads calibration data, processes stereo pairs, applies rotation corrections, performs feature matching, and triangulates the points to reconstruct the 3D structure of the kite.

Key Steps:

Stereo Calibration: Loads camera calibration data for intrinsic and extrinsic parameters.

Rotation Corrections: Uses per-frame yaw, pitch, and roll adjustments to correct for camera misalignments.

Feature Matching: Matches corresponding features in the left and right frames using epipolar constraints and KLT tracking.

3D Triangulation: Computes the 3D coordinates of the matched features.

Purpose: This script ties together all processes from calibration, feature matching, and triangulation, generating the final 3D point cloud used in analysis.

4. stereo_photogrammetry_utils.py

Function: This script includes various utilities for stereo image processing, such as feature tracking, rectification, and triangulation. It also handles stereo pair processing with rotation corrections.

Key Features:

Stereo Rectification: Adjusts and aligns stereo frames to correct for distortions and ensure epipolar geometry for accurate feature matching.

KLT Tracking: Uses Lucas-Kanade optical flow to track points across frames.

Triangulation: Converts matched 2D points from both stereo images into 3D coordinates.

Debugging and Visualization: Offers options for visualizing stereo frames with overlaid markers and diagnostic statistics, such as epipolar line alignment.

Purpose: It provides the core processing functions for handling stereo images, including rectification, tracking, and 3D reconstruction, enabling precise measurement of the kite's deformation.

**UWB subsystem**

The UWB subsystem handles the setup, calibration, and data collection of UWB distance measurements. It also provides tools for post-processing and visualizing the data. This subsystem is integral for measuring distances in the photogrammetry pipeline, where accurate positioning data is required for 3D reconstruction.

1. Calibration Folder:

calibration_UWB.py

Function: This script performs linear calibration of the UWB sensors by fitting a linear model to the raw and actual distance measurements. It calculates the coefficients (a, b) for the calibration equation: corrected = a * measured + b.

Purpose: Provides the calibration parameters to adjust raw UWB measurements and reduce errors.

ranging_accuracy_uwb.py

Function: This script analyzes the noise in UWB measurements by determining the consistency and accuracy of the sensor readings over time.

Purpose: Helps assess the precision of the UWB system by measuring how much variation exists in the readings over a short period (typically 5 seconds), ensuring stable measurements for later use.

static_measurement.py

Function: Takes UWB distance measurements over a fixed 5-second period and calculates the average, since static setups (no movement) are assumed for this test.

Purpose: Provides a stable and reliable UWB reading for calibration and further analysis.

range_test.py

Function: Determines the range of the UWB sensors by measuring the distance over a longer duration (typically 1 minute), simulating the conditions where one device is moved away from the other.

Purpose: Provides an understanding of the operational range and signal loss for the UWB system as the distance between sensors increases.

fresnel_zone_calculation.py

Function: Calculates the Fresnel zone radius, which indicates the minimum ground clearance required to avoid obstruction of the UWB signal. This is based on the operating frequency and the separation distance between the two devices.

Purpose: Ensures that the UWB sensors are positioned at the appropriate height to avoid signal degradation due to ground interference.

2. Data Processing Folder:

lag_light_wrt_first_log.py

Function: Measures the lag between the UWB signal and the LED light used for synchronization. It compares timestamps between UWB logs and the light flashes (triggered by the system) to determine the time offset between them.

Purpose: Ensures synchronization between the UWB system and external signals (e.g., light flashes) to maintain precise time alignment in the data.

noise_visualization_uwb.py

Function: This script visualizes the UWB distance noise by plotting the variations in the UWB data. It applies a zero-phase exponential moving average (EMA) filter to the data to highlight slow variations and suppress high-frequency noise without introducing any time lag.

Purpose: Provides a visual representation of the noise characteristics, helping assess the quality of UWB data and guiding improvements.

postprocess_RAW_UWB_file.py

Function: This script post-processes raw UWB data that has not been automatically processed. It applies the previously calculated linear calibration, interpolates missing data, and smooths the data using the zero-phase EMA filter.

Purpose: Ensures the raw UWB data is cleaned and calibrated, ready for further analysis.

3. Main UWB Code:

main_UWB.py

Function: This is the main script that initiates UWB ranging. It starts the distance measurement process, logs the data, applies calibration, and post-processes the results. It interacts with the UWB sensors and automatically saves the processed data for later use.

Purpose: The central code for UWB ranging in the pipeline, ensuring that data is collected, processed, and stored automatically.

**Main postprocessing pipeline**

This pipeline integrates the results from the KCU + Pitot, Photogrammetry, and UWB subsystems and generates a synchronized, processed output. It involves combining the data into a single dataset and visualizing it as a video with relevant metrics overlaid.

1. combine_all_results.py

Function: Combines the results from the KCU + Pitot, Photogrammetry, and UWB subsystems into a single, unified Pandas DataFrame. This is done by merging the data based on timestamps or other relevant parameters from each subsystem (e.g., camera frames, sensor data).

Purpose: This script consolidates all collected data from different subsystems, making it ready for further processing and visualization.

2. video_generator.py

Function: This script generates a video from the synchronized and combined data. The video includes:

4 views of the kite: front, side, bottom, and 3D.

Phase of flight and other relevant telemetry data are displayed on the video.

Synchronized data (e.g., wind speed, tether force) is overlaid on the video in real-time.

Key Steps:

Data Visualization: The telemetry data (e.g., airspeed, tether force, span) is used to generate annotations that are overlaid on the video frames.

Rendering 3D Views: The 3D structure of the kite, including markers and the leading edge (LE), is visualized in 3D.

Panels: Different panels show the front, side, bottom views of the kite, as well as metrics like angle of attack (AOA) and tether length.

Output: The final video is saved, providing a comprehensive visual representation of the kite's performance during the test.

Purpose of the Main Post-Processing Pipeline:

The pipeline provides a comprehensive, synchronized output of the data collected from different sensors and cameras. It not only combines these results into a unified dataset but also creates a visual representation in the form of a video. This allows for easy analysis of the kite's behavior, deformation, and performance over time, with all the necessary telemetry overlaid in a clear format.
