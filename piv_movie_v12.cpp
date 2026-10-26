/* ==================================================================================== */
/*  piv_movie_v12.cpp  --  PIV-like velocimetry on schlieren / shadowgraph videos        */
/*                                                                                      */
/*  Derived from piv_movie_v11.cpp. Changes with respect to v11:                        */
/*   1. Pre-processing of the frames before motion filter and PIV (PREPROC):            */
/*        0 none | 1 spatial high-pass | 2 temporal-median background subtraction |     */
/*        3 background subtraction followed by high-pass                                */
/*   2. Motion filter: the normalized phi_bar is now used EVERYWHERE (motion count and  */
/*      PIV mask). v11 counted cells with the raw phi (gray levels) against a threshold */
/*      meant for phi_bar.                                                              */
/*   3. Matching metric selectable (MATCH_MODE): 0 = SQDIFF (as v11), 1 = CCOEFF_NORMED */
/*      (zero-mean normalized cross-correlation, peak = maximum). Optional sub-pixel    */
/*      peak fit (SUBPIXEL) and correlation-peak rejection (CORR_MIN).                  */
/*   4. Bug fixes: row stride of the cell index (was nPIVcellsy, now nPIVcellsx);       */
/*      data race on the motion counter; out-frames/out-motion were appended to instead */
/*      of overwritten; template/image arguments of matchTemplate in the right order;   */
/*      stack-allocated variable-length arrays replaced by std::vector;                 */
/*      frames are read sequentially (no seek per frame).                               */
/*   5. Near-duplicate frame pairs (mean |diff| < DUP_THRESH gray levels) are flagged   */
/*      in out-dup.dat and not processed (u=v=0, motion count repeated from the last    */
/*      valid frame). {phi_bar is meaningless?}   */
/*                                                                                      */
/*  Output files (in OUT_DIR):                                                          */
/*     out-x.dat, out-y.dat           grid of cell centers (pixels, y pointing up)      */
/*     out-u_FFFF.dat, out-v_FFFF.dat displacement in pixels per (piv_interframe) frame */
/*                                    (now floating point)                              */
/*     out-phi_FFFF.dat               normalized motion filter phi_bar                  */
/*     out-corr_FFFF.dat              correlation peak value (MATCH_MODE 1; 0 otherwise) */
/*     out-frames.dat, out-motion.dat, out-dup.dat   one line per processed frame       */
/*                                                                                      */
/*  => Usage: ./piv_v12 [key=value ...]   e.g.  ./piv_v12 in=video.avi out=res first=1984 */
/*         last=1995 preproc=3 match=1 subpixel=1 thr=1.4                               */
/* ==================================================================================== */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>

#include "opencv2/opencv.hpp"
#include "tbb/parallel_for.h"
#include "tbb/global_control.h"

using namespace std;
using namespace cv;

//PREAMBLE


/* Global parameters (can all be overridden from the command line, see the usage above) */
string in_video = "processed/out_exp10_if1.avi";
string out_dir  = "results/out_exp10_if1";

int first_frame = 0;            // First frame to analyze
int last_frame  = 6000;         // Last frame to analyze
int every_frame = 1;            // Step between analyzed frames

int   wROI             = 32;    // Size of the PIV ROI (mask) in pixels
int   ncellsx          = 100;   // dx = image height / ncellsx  (100 -> 7 px at 768 px, as in the report)
float motion_threshold = 1.4;   // Threshold on the NORMALIZED motion filter phi_bar
int   piv_interframe   = 1;     // Frame gap used for PIV
int   max_threads      = 2;     // Maximum number of threads

/* Pre-processing */
int    preproc     = 1;         // 0 none | 1 high-pass | 2 background | 3 background + high-pass
double hp_sigma    = 15.0;      // Gaussian sigma (pixels) of the low-pass removed by the high-pass
double pre_blur    = 0.0;       // Optional Gaussian smoothing (sigma, pixels) applied after the above
int    bg_nsamples = 100;       // Frames sampled (evenly over the video) for the temporal median

/* Matching */
int   match_mode = 1;           // 0 = SQDIFF (min), 1 = CCOEFF_NORMED (max)
int   subpixel   = 1;           // 1 = sub-pixel peak fit (Gaussian, parabolic fallback)
float corr_min   = 0.0;         // Reject vectors whose correlation peak is below this (MATCH_MODE 1)

int   min_blob   = 30;          // Remove connected motion regions smaller than this many cells (0 = off)
float dup_thresh = 0.3;         // Mean |frame difference| (gray levels) under which a pair is a duplicate

/* Debug output of the pre-processed first frame of each pair as PNG (0 = off) */
int   dump_pre   = 0;
float dump_gain  = 8.0;

/* Function declarations */
struct PIVResult { float u, v, corr; };
Mat   preprocess(const Mat& rawgray, const Mat& bg);
Mat   computeBackground(VideoCapture& cap, int nframes);
float getMotion(const Mat& p1, const Mat& p2, int x0, int y0);
PIVResult PIV(const Mat& p1, const Mat& p2, int x0, int y0);


//HELPERS


void parseArgs(int argc, char** argv){
  for (int i = 1; i < argc; i++){
    string a = argv[i];
    size_t eq = a.find('=');
    if (eq == string::npos){ printf("Ignoring argument '%s' (expected key=value)\n", a.c_str()); continue; }
    string k = a.substr(0, eq), v = a.substr(eq + 1);
    if      (k == "in")        in_video = v;
    else if (k == "out")       out_dir = v;
    else if (k == "first")     first_frame = atoi(v.c_str());
    else if (k == "last")      last_frame = atoi(v.c_str());
    else if (k == "every")     every_frame = atoi(v.c_str());
    else if (k == "wroi")      wROI = atoi(v.c_str());
    else if (k == "ncellsx")   ncellsx = atoi(v.c_str());
    else if (k == "thr")       motion_threshold = atof(v.c_str());
    else if (k == "interframe")piv_interframe = atoi(v.c_str());
    else if (k == "threads")   max_threads = atoi(v.c_str());
    else if (k == "preproc")   preproc = atoi(v.c_str());
    else if (k == "hp_sigma")  hp_sigma = atof(v.c_str());
    else if (k == "pre_blur")  pre_blur = atof(v.c_str());
    else if (k == "bg_n")      bg_nsamples = atoi(v.c_str());
    else if (k == "match")     match_mode = atoi(v.c_str());
    else if (k == "subpixel")  subpixel = atoi(v.c_str());
    else if (k == "corr_min")  corr_min = atof(v.c_str());
    else if (k == "dup_thresh")dup_thresh = atof(v.c_str());
    else if (k == "min_blob")  min_blob = atoi(v.c_str());
    else if (k == "dump_pre")  dump_pre = atoi(v.c_str());
    else if (k == "dump_gain") dump_gain = atof(v.c_str());
    else printf("Unknown parameter '%s'\n", k.c_str());
  }
}

// Frame source with sequential reading and a tiny cache (raw gray + pre-processed)
struct FrameSource {
  VideoCapture cap;
  int next = 0;
  Mat bg;
  map<int, Mat> raw, proc;

  bool get(int idx, Mat& r, Mat& p){
    auto it = raw.find(idx);
    if (it != raw.end()){ r = it->second; p = proc[idx]; return true; }
    if (idx < next){ cap.set(CAP_PROP_POS_FRAMES, idx); next = idx; }   // only if going backwards
    while (next < idx){ if (!cap.grab()) return false; next++; }
    Mat im;
    if (!cap.read(im) || im.empty()) return false;
    next++;
    Mat g;
    if (im.channels() == 3) cvtColor(im, g, COLOR_BGR2GRAY); else g = im;
    flip(g, g, 0);                      // vertical flip so that y points up (as in v11)
    g.convertTo(g, CV_32F);
    raw[idx]  = g;
    proc[idx] = preprocess(g, bg);
    r = raw[idx]; p = proc[idx];
    return true;
  }
  void evictBelow(int idx){
    for (auto it = raw.begin();  it != raw.end();  ) { if (it->first < idx) it = raw.erase(it);  else ++it; }
    for (auto it = proc.begin(); it != proc.end(); ) { if (it->first < idx) it = proc.erase(it); else ++it; }
  }
};


// MAIN PROGRAM
int main(int argc, char** argv){

  parseArgs(argc, argv);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, max_threads);

  // LOAD VIDEO 
  FrameSource src;
  src.cap.open(in_video);
  if (!src.cap.isOpened()){
    printf("Cannot open the video file at '%s'.\nCheck the filename.\n", in_video.c_str());
    return -1;
  }
  int nframes = (int) src.cap.get(CAP_PROP_FRAME_COUNT);
  int imcols  = (int) src.cap.get(CAP_PROP_FRAME_WIDTH);
  int imrows  = (int) src.cap.get(CAP_PROP_FRAME_HEIGHT);
  int dx = imrows / ncellsx;

  // PRE-CHECKS 
  filesystem::create_directories(out_dir);     // v11 refused to run if it did not exist

  if (last_frame + piv_interframe > nframes - 1){
    last_frame = nframes - 1 - piv_interframe;
    printf("Setting last frame to be %i to stay consistent with number of frames in video (%i)\n", last_frame, nframes);
  }
  if (dx == 0){ printf("Pixel dx is 0. Setting it to 1 instead.\n"); dx = 1; }
  else        { printf("Current pixel-dx is %i.\n", dx); }

  // BACKGROUND (if needed) 
  if (preproc == 2 || preproc == 3){
    printf("Computing temporal-median background from %i frames...\n", bg_nsamples);
    src.bg = computeBackground(src.cap, nframes);
    src.cap.set(CAP_PROP_POS_FRAMES, 0);
    src.next = 0;
  }

  // GET X-Y GRID 
  char fname[400];
  int nPIVcellsx = 0, nPIVcellsy = 0;
  {
    sprintf(fname, "%s/out-x.dat", out_dir.c_str());
    FILE *xFile = fopen(fname, "w");
    sprintf(fname, "%s/out-y.dat", out_dir.c_str());
    FILE *yFile = fopen(fname, "w");

    int ROItoplefty = wROI;
    while (ROItoplefty < imrows - 2*wROI){
      int ROItopleftx = wROI;
      nPIVcellsx = 0;
      while (ROItopleftx < imcols - 2*wROI){
        fprintf(xFile, "%i\t", ROItopleftx + wROI/2);
        fprintf(yFile, "%i\t", ROItoplefty + wROI/2);
        ROItopleftx += dx;
        nPIVcellsx++;
      }
      fprintf(xFile, "\n");
      fprintf(yFile, "\n");
      ROItoplefty += dx;
      nPIVcellsy++;
    }
    fclose(xFile);
    fclose(yFile);
  }
  printf("PIV grid: %i x %i cells\n", nPIVcellsx, nPIVcellsy);

  // Frame-scalar files are truncated here (v11 appended to old files)
  for (const char* n : {"out-frames.dat", "out-motion.dat", "out-dup.dat"}){
    sprintf(fname, "%s/%s", out_dir.c_str(), n);
    FILE *t = fopen(fname, "w"); fclose(t);
  }

  // LOOP OVER VIDEO FRAMES FOR PIV
  const int ncells = nPIVcellsx * nPIVcellsy;
  vector<float> phi(ncells), u(ncells), v(ncells), corr(ncells);
  int last_motion_count = 0;

  for (int f = first_frame; f <= last_frame; f += every_frame){

    printf("Processing frame %i/%i\n", f, last_frame);

    Mat raw1, raw2, p1, p2;
    if (!src.get(f, raw1, p1) || !src.get(f + piv_interframe, raw2, p2)){
      printf("One or both image is not properly loaded (frame %i).\n", f);
      return -1;
    }
    src.evictBelow(f + 1);

    bool is_dup = (mean(abs(raw2 - raw1))[0] < dup_thresh);

    if (dump_pre){
      Mat o = p1 * dump_gain + 128.0f;
      o.convertTo(o, CV_8U);                         // saturating conversion
      sprintf(fname, "%s/out-pre_%04d.png", out_dir.c_str(), f);
      imwrite(fname, o);
    }

    // FIND FLOW MOTION USING MOTION FILTER 
    vector<int> rowcount(nPIVcellsy, 0);

    tbb::parallel_for(0, nPIVcellsy, [&](int i){
      for (int j = 0; j < nPIVcellsx; j++){
        int y0 = wROI + i*dx;
        int x0 = wROI + j*dx;
        phi[i * nPIVcellsx + j] = getMotion(p1, p2, x0, y0);          // fixed stride
      }
    });

    // Normalize phi by its average value over the whole frame
    double sumphi = 0;
    for (int k = 0; k < ncells; k++) sumphi += phi[k];
    double avphi = sumphi / ncells;
    if (avphi > 1e-9){ for (int k = 0; k < ncells; k++) phi[k] = (float)(phi[k] / avphi); }

    // Motion mask: phi_bar >= threshold, optionally with small isolated regions removed.
    // The motion count and the PIV mask below both use this
    vector<uchar> moving(ncells, 0);
    {
      Mat m(nPIVcellsy, nPIVcellsx, CV_8U);
      for (int k = 0; k < ncells; k++) m.data[k] = (phi[k] >= motion_threshold) ? 255 : 0;
      if (min_blob > 0){
        Mat labels, stats, cent;
        connectedComponentsWithStats(m, labels, stats, cent, 8, CV_32S);
        for (int k = 0; k < ncells; k++){
          int l = labels.at<int>(k / nPIVcellsx, k % nPIVcellsx);
          moving[k] = (l > 0 && stats.at<int>(l, CC_STAT_AREA) >= min_blob) ? 1 : 0;
        }
      }
      else { for (int k = 0; k < ncells; k++) moving[k] = (m.data[k] != 0); }
    }
    int motion_count = 0;
    for (int k = 0; k < ncells; k++) motion_count += moving[k];

    // PIV 
    fill(u.begin(), u.end(), 0.f);
    fill(v.begin(), v.end(), 0.f);
    fill(corr.begin(), corr.end(), 0.f);

    if (is_dup){
      motion_count = last_motion_count;       // keep the jet-size curve continuous, flagged in out-dup.dat
    }
    else {
      tbb::parallel_for(0, nPIVcellsy, [&](int i){
        for (int j = 0; j < nPIVcellsx; j++){
          int k = i * nPIVcellsx + j;
          if (!moving[k]) continue;                                   // no motion: u = v = 0
          int y0 = wROI + i*dx;
          int x0 = wROI + j*dx;
          PIVResult r = PIV(p1, p2, x0, y0);
          corr[k] = r.corr;
          if (match_mode == 1 && r.corr < corr_min) continue;         // rejected vector
          u[k] = r.u;
          v[k] = r.v;
        }
      });
      last_motion_count = motion_count;
    }

    // WRITE TO FILE  
    FILE *uFile, *vFile, *phiFile, *cFile;
    sprintf(fname, "%s/out-u_%04d.dat", out_dir.c_str(), f);    uFile   = fopen(fname, "w");
    sprintf(fname, "%s/out-v_%04d.dat", out_dir.c_str(), f);    vFile   = fopen(fname, "w");
    sprintf(fname, "%s/out-phi_%04d.dat", out_dir.c_str(), f);  phiFile = fopen(fname, "w");
    sprintf(fname, "%s/out-corr_%04d.dat", out_dir.c_str(), f); cFile   = fopen(fname, "w");

    for (int i = 0; i < nPIVcellsy; ++i){
      for (int j = 0; j < nPIVcellsx; ++j){
        int k = i * nPIVcellsx + j;
        fprintf(uFile,   "%8.3f\t", u[k]);
        fprintf(vFile,   "%8.3f\t", v[k]);
        fprintf(phiFile, "%6.3f\t", phi[k]);
        fprintf(cFile,   "%6.3f\t", corr[k]);
      }
      fprintf(uFile, "\n"); fprintf(vFile, "\n"); fprintf(phiFile, "\n"); fprintf(cFile, "\n");
    }
    fclose(uFile); fclose(vFile); fclose(phiFile); fclose(cFile);

    sprintf(fname, "%s/out-frames.dat", out_dir.c_str());
    FILE *framesFile = fopen(fname, "a"); fprintf(framesFile, "%i\n", f); fclose(framesFile);
    sprintf(fname, "%s/out-motion.dat", out_dir.c_str());
    FILE *motionFile = fopen(fname, "a"); fprintf(motionFile, "%i\n", motion_count); fclose(motionFile);
    sprintf(fname, "%s/out-dup.dat", out_dir.c_str());
    FILE *dupFile = fopen(fname, "a"); fprintf(dupFile, "%i\n", is_dup ? 1 : 0); fclose(dupFile);
  }

  return 0;
}


// USER DEFINED FUNCTIONS 

/*
  computeBackground(cap, nframes)
    Per-pixel temporal median of bg_nsamples frames sampled evenly over the whole video.
    Static structures (vignetting, sensor seam, dust, optics) are in it; a jet that is only
    present in a minority of the frames is not.
*/
Mat computeBackground(VideoCapture& cap, int nframes){
  vector<Mat> samples;
  for (int s = 0; s < bg_nsamples; s++){
    int idx = (int)((s + 0.5) * nframes / bg_nsamples);
    cap.set(CAP_PROP_POS_FRAMES, idx);
    Mat im, g;
    if (!cap.read(im) || im.empty()) continue;
    if (im.channels() == 3) cvtColor(im, g, COLOR_BGR2GRAY); else g = im;
    flip(g, g, 0);
    samples.push_back(g);
  }
  int n = (int) samples.size();
  Mat bg(samples[0].size(), CV_32F);
  vector<uchar> vals(n);
  for (int r = 0; r < bg.rows; r++){
    for (int c = 0; c < bg.cols; c++){
      for (int s = 0; s < n; s++) vals[s] = samples[s].at<uchar>(r, c);
      nth_element(vals.begin(), vals.begin() + n/2, vals.end());
      bg.at<float>(r, c) = (float) vals[n/2];
    }
  }
  return bg;
}

/*
  preprocess(rawgray, bg)
    Returns the float image that is given to the motion filter and to PIV.
*/
Mat preprocess(const Mat& rawgray, const Mat& bg){
  Mat g = rawgray.clone();
  if (preproc == 2 || preproc == 3) g = g - bg;
  if (preproc == 1 || preproc == 3){
    Mat low;
    GaussianBlur(g, low, Size(0, 0), hp_sigma, hp_sigma, BORDER_REFLECT);
    g = g - low;
  }
  if (pre_blur > 0) GaussianBlur(g, g, Size(0, 0), pre_blur, pre_blur, BORDER_REFLECT);
  return g;
}

/*
  getMotion(p1, p2, x0, y0)
    Motion filter: RMS gray-level difference between the two images in the ROI whose top-left
    corner is (x0, y0).
*/
float getMotion(const Mat& p1, const Mat& p2, int x0, int y0){
  Mat a = p1(Range(y0, y0 + wROI), Range(x0, x0 + wROI));
  Mat b = p2(Range(y0, y0 + wROI), Range(x0, x0 + wROI));
  return (float)(norm(b, a, NORM_L2) / wROI);          // sqrt(sum (b-a)^2 / wROI^2)
}

/*
  PIV(p1, p2, x0, y0)
    Template matching of the wROI x wROI mask of image 1 inside the (3 wROI) x (3 wROI)
    seeking region of image 2. Returns the displacement (u, v) in pixels and the correlation
    peak value.
*/
PIVResult PIV(const Mat& p1, const Mat& p2, int x0, int y0){
  PIVResult out = {0.f, 0.f, 0.f};

  Mat tmpl = p1(Range(y0, y0 + wROI), Range(x0, x0 + wROI));
  Mat ext  = p2(Range(y0 - wROI, y0 + 2*wROI), Range(x0 - wROI, x0 + 2*wROI));

  Mat res;
  matchTemplate(ext, tmpl, res, match_mode == 0 ? TM_SQDIFF : TM_CCOEFF_NORMED);   // image first, template second

  double minVal, maxVal;
  Point minLoc, maxLoc;
  minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

  Point pk = (match_mode == 0) ? minLoc : maxLoc;
  double pv = (match_mode == 0) ? minVal : maxVal;
  if (!std::isfinite(pv)) return out;                 // flat (zero-variance) window

  double dxs = 0, dys = 0;
  if (subpixel && pk.x >= 1 && pk.y >= 1 && pk.x <= res.cols - 2 && pk.y <= res.rows - 2){
    auto fit = [&](double a, double c, double b) -> double {
      // a, c, b = values at -1, 0, +1 around the integer peak
      if (match_mode == 1 && a > 0 && c > 0 && b > 0){            // Gaussian 3-point fit
        double la = log(a), lc = log(c), lb = log(b);
        double den = 2.0 * (la - 2.0*lc + lb);
        if (den < 0) return (la - lb) / den;
      }
      double den = 2.0 * (a - 2.0*c + b);                         // parabolic fit
      if (fabs(den) > 1e-12) { double d = (a - b) / den; return (fabs(d) <= 0.5) ? d : 0.0; }
      return 0.0;
    };
    dxs = fit(res.at<float>(pk.y, pk.x - 1), res.at<float>(pk.y, pk.x), res.at<float>(pk.y, pk.x + 1));
    dys = fit(res.at<float>(pk.y - 1, pk.x), res.at<float>(pk.y, pk.x), res.at<float>(pk.y + 1, pk.x));
    if (fabs(dxs) > 0.5) dxs = 0;                                  // Gaussian fit safety
    if (fabs(dys) > 0.5) dys = 0;
  }

  out.u = (float)(pk.x + dxs - wROI);
  out.v = (float)(pk.y + dys - wROI);
  out.corr = (match_mode == 1) ? (float) pv : 0.f;
  return out;
}
