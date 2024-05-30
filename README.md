# E2E using SPIN

The `yolov7` and `spin` and `gpu` folder are cloned from their github repos and are heavily changed to run E2E flow.

Install appropriate conda environment for [`yolov7`](https://github.com/WongKinYiu/yolov7), [`spin`](https://github.com/nkolot/SPIN) and [`gpu`](https://github.com/HynDufO/xpbd-vto). 

Install `pip install exec-wrappers`.
Create wrappers for `yolov7` and `spin` conda. This create `/tmp/yolov7` and `/tmp/spin` (Check `run_yolov7.sh` and `run_spin.sh` for more details)
```
create-wrappers  -t conda --bin-dir ~/.conda/envs/yolov7/bin --dest-dir /tmp/yolov7 --conda-env-dir ~/.conda/envs/yolov7
create-wrappers  -t conda --bin-dir ~/.conda/envs/spin/bin --dest-dir /tmp/spin --conda-env-dir ~/.conda/envs/spin
```
Activate the conda environment for warp and cd to the gpu folder. Then run something like this:
```
/path/to/run_yolov7.sh | /path/to/run_spin.sh | python gpu_combine.py
```

