import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import sys

# USER INTERFACE =========================================================================

# Data to plot
in_dir = "results/exp3_tests2"
iframe = 4022
wROI   = 32
motion_threshold = 29

# Set video settings
ref_vid = "videos/out_exp3_IF-4_full_mjpeg_HQ.avi"
xscale = 2230    # pix/m
tscale = 400     # frames/sec

# Plot options
mot_av_width = 0
hist_av_width = 1
hist_nbins = 100


# VIDEO SETTINGS =========================================================================

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

# Load video and frame
cap = cv2.VideoCapture(ref_vid)
cap.set(1, iframe)
ret, im1 = cap.read()
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im1 = cv2.flip(im1, 0)

# PRE-PROCESS ============================================================================

# Load data
frames = np.loadtxt(in_dir+"/out-frames.dat")
motion = np.loadtxt(in_dir+"/out-motion.dat")

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

# Sliding-average on motion pixel count
frames_av = np.zeros(frames.size - 2*mot_av_width)
motion_av = np.zeros(frames.size - 2*mot_av_width)

if mot_av_width != 0:
	i = 0
	j = mot_av_width
	while (j+mot_av_width < frames.size):
		frames_av[i] = frames[j]
		motion_av[i] = np.average( motion[j-mot_av_width : j+mot_av_width] )
		
		i = i+1
		j = j+1

else:
	frames_av = frames
	motion_av = motion
	
	
# Get global jet expansion rate (pix/frame)
Vjet = np.gradient(motion_av, frames_av)

# Get NumPy histogram
h, b0 = np.histogram(Vmag_dim, hist_nbins)
b = np.zeros(h.size)
for i in range(b.size):
	b[i] = (b0[i] + b0[i+1])/2

# Make the histogram smooth
b_av = np.zeros(b.size - 2*hist_av_width)
h_av = np.zeros(b.size - 2*hist_av_width)

if hist_av_width != 0:
	i = 0
	j = hist_av_width
	while (j+hist_av_width < b.size):
		b_av[i] = b[j]
		h_av[i] = np.average( h[j-hist_av_width : j+hist_av_width] )
	
		i = i+1
		j = j+1

else:
	b_av = b
	h_av = h

# PLOTS ==================================================================================

'''
# Vector field / streamlines
plt.figure(1)
plt.clf()
plt.pcolormesh(im1, cmap="gray", alpha=0.5)
plt.quiver(X, Y, Unorm, Vnorm, Vmag_dim, norm=colors.PowerNorm(gamma=0.5))
#plt.streamplot(X, Y, Unorm, Vnorm, density=4, color="r", linewidth=0.7, arrowsize = 0.8)
plt.colorbar()
plt.axis("equal")
plt.show()
'''

# Motion filter phi
plt.figure(2)
plt.clf()

plt.pcolormesh(im1, cmap="gray", alpha=0.5)
#ctr = plt.contour(X, Y, phi, range(20,40,4))
ctr = plt.contour(X, Y, phi, 20)
plt.colorbar()
plt.clabel(ctr, ctr.levels, fmt="%.1f", fontsize=8)
ctr = plt.contour(X, Y, phi, (motion_threshold,), colors="k", linewidths=2)
plt.axis("equal")
plt.show()
'''
# Velocity magnitude
plt.figure(3)
plt.clf()
plt.pcolormesh(im1, cmap="gray", alpha=1.0)
plt.pcolormesh(X, Y, Vmag_dim, alpha=0.2, norm=colors.PowerNorm(gamma=0.5))
plt.colorbar()
ctr = plt.contour(X, Y, phi, (motion_threshold,), colors="k", linewidths=2)
plt.axis("equal")
plt.show()

# Velocity histogram
plt.figure(4)
plt.clf()
plt.hist(histVmag_dim, bins=hist_nbins, range=(0.0, Vmax_dim), color="k")
plt.plot(b_av, h_av, "c-o")
ax = plt.gca()
ax.set_yscale("log")
plt.ylim(1, npoints)

# Jet surface
plt.figure(5)
#plt.clf()
plt.plot(frames_av/tscale, motion_av, "b")
plt.axvline(iframe/tscale, c="r")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel("Time (s)")
plt.ylabel("Count of pixel regions in motion (pix)")
plt.tight_layout()
plt.show()


# Jet expansion rate
plt.figure(6)
#plt.clf()
plt.plot(frames_av/tscale, Vjet, "b")
plt.axvline(iframe/tscale, c="r")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel("Time (s)")
plt.ylabel("Jet expansion rate (pix/frame)")
plt.tight_layout()
plt.show()

# Alternate (smooth) histogram)
plt.figure(7)
plt.clf()
plt.semilogy(b_av, h_av, "k-o")
plt.ylim(1, npoints)
plt.show()
sys.exit()
'''



