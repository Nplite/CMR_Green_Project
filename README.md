# DeepStream dGPU Setup for Ubuntu

## Prerequisites

Install the following packages:

```
sudo pip3 install meson
sudo pip3 install ninja
```

## Compilation and Installation of GLib
1. Clone the GLib repository and navigate into it:

   ```
   git clone https://github.com/GNOME/glib.git
   cd glib
   ```
2. Build and install GLib:

   ```
   meson build --prefix=/usr
   ninja -C build/
   cd build/
   ninja install
   ```

## Install Dependencies:

```
sudo apt install \
libssl3 \
libssl-dev \
libgles2-mesa-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev \
libjsoncpp-dev \
protobuf-compiler \
gcc \
make \
git \
python3
```

## Install CUDA Toolkit 12.6

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6
```


## Install NVIDIA driver 560.35.03 (for RTX GPUs)

```
chmod 755 NVIDIA-Linux-x86_64-560.35.03.run
sudo ./NVIDIA-Linux-x86_64-560.35.03.run --no-cc-version-check
```

## Install TensorRT 10.6.0.26

```
version="10.6.0.26-1+cuda12.6"
sudo apt-get install libnvinfer-dev=${version} libnvinfer-dispatch-dev=${version} libnvinfer-dispatch10=${version} libnvinfer-headers-dev=${version}
libnvinfer-headers-plugin-dev=${version} libnvinfer-lean-dev=${version} libnvinfer-lean10=${version} libnvinfer-plugin-dev=${version} libnvinfer-plugin10=${version}
libnvinfer-vc-plugin-dev=${version} libnvinfer-vc-plugin10=${version} libnvinfer10=${version} libnvonnxparsers-dev=${version} libnvonnxparsers10=${version} tensorrt-dev=${version}
```

## Install librdkafka (to enable Kafka protocol adaptor for message broker)

1. Clone the librdkafka repository from GitHub:

   ```
   git clone https://github.com/confluentinc/librdkafka.git
   ```

2. Configure and build the library:

    ```
    cd librdkafka
    git checkout tags/v2.2.0
    ./configure --enable-ssl
    make
    sudo make install
    ```
3. Copy the generated libraries to the deepstream directory:

   ```
   sudo mkdir -p /opt/nvidia/deepstream/deepstream/lib
   sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream/lib
   sudo ldconfig
   ```

## Install the DeepStream SDK
Download the DeepStream 7.1 dGPU Debian package deepstream-7.1_7.1.0-1_amd64.deb : https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream
Enter the command:
```
sudo apt-get install ./deepstream-7.1_7.1.0-1_amd64.deb
```
## Verification: 
##### Once DeepStream SDK installation is successful, refer to [Expected output (deepstream-app)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#ds-quickstart-ds-app-x86) for the expected output.


## DeepStream python bindings
## 1.1 Base dependencies
```
sudo apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev
```
##### Additionally, install PyPA :
```
pip3 install build
```

## 1.2 Initialization of submodules
Make sure you clone the deepstream_python_apps repo under <DeepStream ROOT>/sources: git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

This will create the following directory:
```
<DeepStream ROOT>/sources/deepstream_python_apps
```
The repository utilizes gst-python and pybind11 submodules. To initializes them, run the following command:
```
cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
git submodule update --init
python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse
```

## 1.3 Installing Gst-python
Build and install gst-python:
```
cd bindings/3rdparty/gstreamer/subprojects/gst-python/
meson setup build
cd build
ninja
ninja install
```

## 2. Compiling the bindings
Python bindings are compiled using PyPA with a CMake extension. The compiled bindings wheel will be found inside directory dist/. Following commands provide instructions for common compilation options:

## 2.1 Quick build 1 (x86-ubuntu-22.04 | python 3.10 | Deepstream 7.1)
```
cd deepstream_python_apps/bindings
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
python3 -m build
```
## 2.2 Build DeepStreamSDK python bindings
After the container launches successfully, while inside the cross-compile docker, run following commands:
```
# cd to /opt/nvidia/deepstream/deepstream/sources/ dir
cd /opt/nvidia/deepstream/deepstream/sources/

# Sync deepstream_python_apps repo from github
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

cd deepstream_python_apps/bindings/

# Initialize submodules
git submodule update --init

# Set parallelization level
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# Build pybind wheel
python3 -m build

# Copy the pyds wheel package to the export dir
cp dist/pyds-*.whl /export_pyds
```

If you encounter an error regarding a missing file libnvds_osd.os at /opt/nvidia/deepstream/deepstream/lib/libnvds_osd.so during ```python3 -m build```, follow these steps:

1. Navigate to the CMakeLists.txt file in the deepstream_python_apps/bindings directory.
2. Locate the line:
   ```
   check_variable_set(DS_PATH "/opt/nvidia/deepstream/deepstream")
   ```
3. Modify it to:
   ```
   check_variable_set(DS_PATH "/opt/nvidia/deepstream/deepstream-7.1")
   ```


## 3. Installing the bindings
Following commands can be used to install the generated pip wheel.

## 3.1 Installing the pip wheel
```
cd dist/
pip3 install ./pyds-1.2.0-*.whl
```
## 3.2 Launching test-1 app

```
cd apps/deepstream-test1
python3 deepstream_test_1.py <input .h264 file>
```
