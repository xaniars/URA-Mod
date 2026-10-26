import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import sys

from scipy.ndimage import gaussian_filter


# USER INTERFACE =========================================================================

# Set video settings
ref_vid = "videos/out_exp5_IF-4_full_mjpeg_HQ.avi"
xscale = 2230    # pix/m
tscale = 400     # frames/sec

# Data to plot
in_dir = "results/exp5_ROI-32_full"
first_frame = 0
last_frame  = 7995
every_frame = 10
wROI = 32

# Plot options
hist_nbins = 200
hist_av_width = 5

sigma_x = 0.0
sigma_y = 0.0

# Output options
out_dir = "piv_movies/exp5_ROI-32_full"
im_name = "im_"
show_last_plot = False

# START SCRIPT HERE ======================================================================

# Load video
cap = cv2.VideoCapture(ref_vid)

# Load X-Y grid
X = np.loadtxt(in_dir+"/out-x.dat")
Y = np.loadtxt(in_dir+"/out-y.dat")

# Get dimensions of the grid
nx = X.shape[1]
ny = X.shape[0]
npoints = nx*ny
nframes = last_frame - first_frame

# GENERATE FRAMES ========================================================================

# Prepare tables
f = np.arange(first_frame, last_frame, every_frame)
phi = np.zeros((f.size, ny, nx))

# Loop over all data to get velocity histogram
iframe = first_frame
i = 0
while (iframe < last_frame):
	
	# Print advancement
	print("Generating histogram for frame %i/%i" %(iframe, last_frame))
	
	# Load data
	phi[i] = np.loadtxt(in_dir+"/out-phi_%04d.dat" %(iframe))
	
	iframe = iframe + every_frame
	i = i+1

# PLOT DATA ==========================================================================

def plot_phi(i, contour):

	# Load video and frame
	cap = cv2.VideoCapture(ref_vid)
	cap.set(1, f[i])
	ret, im1 = cap.read()
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im1 = cv2.flip(im1, 0)
	
	# Motion filter phi
	plt.clf()

	plt.pcolormesh(im1, cmap="gray")
	ctr = plt.contour(X, Y, phi_norm[i], \
						(contour,), colors="r")
						#np.arange(0.2, 0.5, 0.02), norm=colors.PowerNorm(gamma=1.5))
	#plt.colorbar()
	#plt.clabel(ctr, ctr.levels, fmt="%.3f", fontsize=8)
	plt.axis("equal")
	plt.show()

	


















