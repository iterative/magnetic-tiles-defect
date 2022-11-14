sudo apt update -y
sudo apt install -y libdrm-dev libsystemd-dev
sudo apt install -y cmake libncurses5-dev libncursesw5-dev git
git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON -DINTEL_SUPPORT=ON
make
sudo make install
rm -rf nvtop/