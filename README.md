# samurai_tensorrt_py
Samurai使用tensorRT python推理

requirements:
- tensorRT >= 10.0.1
- pycuda = 2025.1
- onnx = 1.17.0

#### 安装 tensorRT

download [tensorrt 10.1.0](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-11.8.tar.gz), 解压后cd 到TensorRT-10.0.1/python目录下，执行
```shell
pip install tensorrt-10.0.1-cp310-none-linux_x86_64.whl
pip install tensorrt_dispatch-10.0.1-cp310-none-linux_x86_64.whl
pip install tensorrt_lean-10.0.1-cp310-none-linux_x86_64.whl
```

#### SAM 2.1 Checkpoint Download
https://github.com/facebookresearch/sam2/tree/main/checkpoints

#### export onnx
to be continued

#### 运行
```shell
python main.py
```