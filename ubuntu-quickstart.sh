#!/bin/sh
# Script to quickly setup an ipython notebook env on a stock Ubuntu 13.04

set -ex


sudo apt-get install -y \
    python-numpy python-scipy python-dev libatlas-dev \
    python-zmq python-pip python-virtualenv \
    git numactl htop

cd ~

if [ ! -d "venv" ]; then
    virtualenv --system-site-packages venv
fi
. venv/bin/activate

pip install scikit-learn ipython[notebook]

git config --global user.name "Olivier Grisel"
git config --global user.email olivier.grisel@ensta.org

if [ ! -x "~/.ssh/config" ]; then
    echo "Host github.com" >> ~/.ssh/config
    echo "    StrictHostKeyChecking no" >> ~/.ssh/config
fi

if [ ! -d "notebooks" ]; then
    git clone git@github.com:ogrisel/notebooks.git
fi

if [ ! -d "/mnt/ubuntu" ]; then
    sudo mkdir /mnt/ubuntu
    sudo chown -R ubuntu. /mnt/ubuntu

    mkdir /mnt/ubuntu/data
    ln -s /mnt/ubuntu/data
fi

# (Re)start the notebook process
cd ~/notebooks
pkill -9 -f "disabled-ipython-browser"
nohup ~/venv/bin/ipython notebook \
    --ip="*" \
    --browser="disabled-ipython-browser" &

