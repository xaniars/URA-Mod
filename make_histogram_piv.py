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
every_frame = 1
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
#cap = cv2.VideoCapture(ref_vid)

# Load X-Y grid
X = np.loadtxt(in_dir+"/out-x.dat")
Y = np.loadtxt(in_dir+"/out-y.dat")

# Get dimensions of the grid
nx = X.shape[1]
ny = X.shape[0]
npoints = nx*ny
nframes = last_frame - first_frame

# Get maximum velocity recordable
Vmax_dim = (2**0.5 * wROI) * tscale/xscale

# GENERATE HISTOGRAM =====================================================================

# Prepare histogram
f = np.arange(first_frame, last_frame, every_frame)
h0 = np.zeros((hist_nbins, f.size))

# Loop over all data to get velocity histogram
iframe = first_frame
i = 0
while (iframe < last_frame):
	
	# Print advancement
	print("Generating histogram for frame %i/%i" %(iframe, last_frame))
	
	# Load data
	U = np.loadtxt(in_dir+"/out-u_%04d.dat" %(iframe))
	V = np.loadtxt(in_dir+"/out-v_%04d.dat" %(iframe))
	
	# Get velocity magnitude (pix/frame) and dimensional
	Vmag = (U**2.0 + V**2.0)**0.5
	Vmag_dim = Vmag * tscale/xscale

	# Get NumPy histogram and make it plotable
	h0[:,i], b0 = np.histogram(Vmag_dim, hist_nbins)
	
	iframe = iframe + every_frame
	i = i+1

	
# PLOT DATA ==========================================================================

# Make the histogram smooth
b = np.zeros( b0.size - 2*hist_av_width)
h = np.zeros((b0.size - 2*hist_av_width , h0.shape[1]))

if hist_av_width != 0:
	
	# Loop on all frames
	for k in range(f.size):	
		
		# Apply average
		i = 0
		j = hist_av_width
		while (j+hist_av_width < b0.size):
			b[i] = b0[j]
			h[i,k] = np.average( h0[j-hist_av_width : j+hist_av_width, k] )
	
			i = i+1
			j = j+1
else:
	b = b0
	h = h0

# Generate meshgrid for plot
F, B = np.meshgrid(f, b)

# Plot histogram history
plt.figure(1)
plt.clf()
plt.pcolormesh(F, B, h, norm=colors.PowerNorm(0.2))
plt.xlabel("Frame (-)")
plt.ylabel("Velocities (m/s)")
plt.tight_layout()
plt.show()
plt.savefig("histogram_history.png", dpi=300)








sys.exit()















