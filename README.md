# simpunch

This software assists in simulating PUNCH-like data. It's useful in testing the pipeline software.

## Instructions
First run, `generate_l3 /Users/jhughes/Desktop/data/gamera_mosaic_jan2024/ 1`.

Then, run the visualize script `python visualize.py`.

Finally, make movies:
```
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PAN_*_total.png' -c:v libx264 -pix_fmt yuv420p PAN_total.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PAN_*_polarized.png' -c:v libx264 -pix_fmt yuv420p PAN_polarized.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PNN_*_total.png' -c:v libx264 -pix_fmt yuv420p PNN_total.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PNN_*_polarized.png' -c:v libx264 -pix_fmt yuv420p PNN_polarized.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PAM_*_total.png' -c:v libx264 -pix_fmt yuv420p PAM_total.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PAM_*_polarized.png' -c:v libx264 -pix_fmt yuv420p PAM_polarized.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PTM_*_total.png' -c:v libx264 -pix_fmt yuv420p PTM_total.mp4
ffmpeg -framerate 10 -pattern_type glob -i '/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3_trial/PUNCH_L3_PTM_*_polarized.png' -c:v libx264 -pix_fmt yuv420p PTM_polarized.mp4
```
