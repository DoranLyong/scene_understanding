# Installation 
```bash
# -- We run on python==3.12 and PyTorch2 with CUDA 12.8  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
```

Required packages 
```bash
pip install -r requirements.txt
```

Chamfer Distance & emd
```
cd ./extensions/chamfer_dist && pip install -e .
cd ./extensions/emd && pip install -e .
```


PointNet++
```
git clone https://github.com/DoranLyong/Pointnet2_PyTorch_Install.git
cd ./pointnet2_ops_lib
pip install -e .
```

GPU kNN
```bash
# -- C++ for miniconda 
conda install -c conda-forge libstdcxx-ng

# -- build 
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA 
make && make install
```

Mamba 
```bash
# -- causal depthwise conv1d
git clone --branch v1.2.0 --single-branch https://github.com/Dao-AILab/causal-conv1d.git
cd ./causal-conv1d
pip install -e .

# -- mamba
cd PointMamba/mamba
pip install -e .
```


