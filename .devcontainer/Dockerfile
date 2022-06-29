FROM mcr.microsoft.com/vscode/devcontainers/python:3.9
ENV DEBIAN_FRONTEND=noninteractive
COPY requirements.txt /tmp/requirements.txt
RUN apt update && apt install -y cmake &&\
	git clone https://github.com/Syllo/nvtop.git &&\
	mkdir -p nvtop/build && cd nvtop/build &&\
	cmake .. &&\
	make &&\
	sudo make install &&\
	pip3 install --upgrade pip &&\
	pip3 install -r /tmp/requirements.txt
