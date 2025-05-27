FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git nano \
      python3 python3-dev python3-pip \
      libx11-6 libxi6 libxrandr2 libxinerama1 libxcursor1 libxrender1 \
      libgl1-mesa-glx libglu1-mesa \
      libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
 && pip install --no-cache-dir --upgrade pip

WORKDIR /workspace

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir \
      torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir -r requirements.txt opencv-python tensorflow

COPY . /workspace/

WORKDIR /workspace
CMD ["bash"]

# Also do that at the first time in the docker container:
# cd models/correlation_package/
# python3 setup.py build_ext --inplace
# cd ../..
# export LD_LIBRARY_PATH=$(python3 - <<<'import os,torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH}
