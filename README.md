# E2E using SPIN

The `yolov7` and `spin` and `gpu` folder are cloned from their github repos and are heavily changed to run E2E flow.

Install appropriate conda environment for yolov7, spin and warp. Activate the conda environment for warp and cd to the gpu folder and run something like this:
```
/path/to/run_yolov7.sh | /path/to/run_spin.sh | python gpu_combine.py
```

