export CUDACXX=/usr/local/cuda-11.7/bin/nvcc
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release
nsys nvprof ./plenoctree-app