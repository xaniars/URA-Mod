#!/bin/bash
ffmpeg -r 20 -i film_temp/frame_%05d.png -f avi -vcodec mjpeg -vb 20M movie.avi
