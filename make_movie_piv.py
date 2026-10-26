import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import sys

# USER INTERFACE =========================================================================

# Set video settings
ref_vid = "exp10_h264.avi"
xscale = 2230    # pix/m
tscale = 400     # frames/sec

# Data to plot
in_dir = "results/exp5_ROI-32_full"
first_frame = 0
last_frame  = 7995
every_frame = 1
wROI = 32

# Plot options
hist_nbins = 100

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

# Get maximum velocity recordable
Vmax_dim = (2**0.5 * wROI) * tscale/xscale

# Create output figure
plt.close("all")
plt.figure(1, (11,6))
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.tight_layout()

# Loop over all data to plot
iframe = first_frame
while (iframe <= last_frame):
	
	# Print advancement
	print("Generating frame %i/%i" %(iframe, last_frame))
	
	# PREPARE DATA =======================================================================
	
	# Load data
	U = np.loadtxt(in_dir+"/out-u_%04d.dat" %(iframe))
	V = np.loadtxt(in_dir+"/out-v_%04d.dat" %(iframe))
	phi = np.loadtxt(in_dir+"/out-phi_%04d.dat" %(iframe))
	
	# Get velocity magnitude (pix/frame)
	Vmag = (U**2.0 + V**2.0)**0.5
	histVmag = np.reshape(Vmag, Vmag.size)
	
	# Get dimensional velocity magnitude (m/s)
	Vmag_dim = Vmag * tscale/xscale
	histVmag_dim = np.reshape(Vmag_dim, Vmag_dim.size)
	histVmag_dim_norm = histVmag_dim / np.amax(histVmag_dim)
	
	# Get normalized velocities for vectorfield plot
	Unorm = U/Vmag
	Vnorm = V/Vmag

	# Load current video frame
	cap.set(1, iframe)
	ret, im1 = cap.read()
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im1 = cv2.flip(im1, 0)
	
	# PLOT DATA ==========================================================================
	
	plt.clf()
	
	# Plot velocity magnitude
	plt.subplot(1,2,1)
	plt.pcolormesh(im1, cmap="gray", alpha=1.0)
	plt.pcolormesh(X, Y, Vmag_dim,\
					vmin=0, vmax=Vmax_dim,\
					alpha=0.2, norm=colors.PowerNorm(gamma=0.5))
	plt.colorbar()
	plt.axis("equal")
	
	# Plot velocity histogram
	plt.subplot(1,2,2)
	plt.hist(histVmag_dim, bins=hist_nbins, range=(0.01, Vmax_dim), color="k")
	ax = plt.gca()
	ax.set_yscale("log")
	plt.ylim(1, npoints)
	
	# Save figure
	plt.tight_layout()
	plt.savefig(out_dir + "/" + im_name + str(iframe).zfill(7) + ".png")

	
	
	# Go to next frame
	iframe = iframe + every_frame

if show_last_plot:
	plt.show()
