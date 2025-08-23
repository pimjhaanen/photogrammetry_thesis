import numpy as np

#Calculate Fresnel Zone for ground clearance, assuming worst case scenario
f_uwb = 6.5 * 10 ** 9
c = 3 * 10 ** 8
wavelength = c/f_uwb

D=50 #max distance kite-ground
d1=d2=D/2

r1=np.sqrt((wavelength*d1*d2)/(D))
print(f"The minimum ground clearance, assuming both sensors at the same height (worst case), should be {round(r1,2)} metres")