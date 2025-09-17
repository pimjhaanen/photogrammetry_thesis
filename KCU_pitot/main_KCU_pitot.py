"""Here should be a code that takes the files from the KCU, and logs UTC, t, u_s, u_p and V_a (and maybe the angles)
Lets say the KCU is started before the UWB, which should be the case. Then the UTC closest to the start UTC of the UWB
should be found, and the corresponding time t should be set to zero. Then the moment of UWB ranging t=0, also on camera's
(the frame should be set to 0) and also on the KCU and pitot.
So it only writes to an output .csv file from the UTC corrsponding to the UWB in the output folder"""

#1) take KCU_pitot file
#2) create a csv file with the columns UTC, t, u_s, u_p and V_a (and maybe the angles)
#3) find the corresponding UWB file, and the first UTC time
#4) that utc time should find its closest match in the KCU_pitot file and find the corresponding timestamp t
#5) this t should be subtracted from all timestamps, meaninng the corresponding utc timestamp is 0
#6) for the duration of the UWB ranging, the timestamps and corresponding data should be put into a pd.dataframe
#7) apply the postprocessing to Va, so calibration coefficients and LPF
