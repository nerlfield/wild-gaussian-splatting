# wild-gaussian-splatting

git clone git@github.com:nerlfield/wild-gaussian-splatting.git --recursive

git pull --recurse-submodules
git submodule update --init --recursive

conda create -n wildgaussians python=3.11 cmake=3.14.0

conda activate wildgaussians 

# General dependencies:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
or
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html # it would be faster

pip install ipywidgets==8.0.2 jupyterlab==3.4.2 lovely-tensors==0.1.15

# Dust3r dependencies:
cd dust3r

pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
pip install -r requirements_optional.txt

# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/

it could took a while:
python setup.py build_ext --inplace
cd ../../../

mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

python3 demo.py --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

cd ..

# Gaussian splatting dependencies:


```sh
# setup pip packages
pip install -r requirements.txt

# setup submodules
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e gaussian-splatting/submodules/simple-knn
```

------


jupyter lab --no-browser --ip 0.0.0.0 --port 4546 --allow-root --notebook-dir=.


python train.py -s ../data/scenes/turtle --white_background