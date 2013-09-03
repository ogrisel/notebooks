#!/bin/sh
# Script to quickly setup an ipython notebook env on a stock Ubuntu 13.04

set -ex


sudo apt-get install -y \
    python-numpy python-scipy python-dev libatlas-dev \
    python-zmq python-pip python-virtualenv \
    git libnuma-dev numactl htop vim python-matplotlib libevent-dev

sudo update-alternatives --set editor /usr/bin/vim.basic

cd $HOME
if [ ! -d "venv" ]; then
    virtualenv --system-site-packages venv
fi
. venv/bin/activate

pip install scikit-learn ipython[notebook] blosc apache-libcloud gevent numa
pip install git+https://github.com/esc/bloscpack

git config --global user.name "Olivier Grisel"
git config --global user.email olivier.grisel@ensta.org

if [ ! -x "$HOME/.ssh/config" ]; then
    if f [ ! -d "$HOME/.ssh" ]; then
        mkdir $HOME/.ssh
    fi
    echo "Host github.com" >> $HOME/.ssh/config
    echo "    StrictHostKeyChecking no" >> $HOME/.ssh/config
fi

if [ ! -d "$HOME/notebooks" ]; then
    git clone git@github.com:ogrisel/notebooks.git
fi

if [ -d "/mnt/resource" ]; then
    # Azure
    DATA_ROOT=/mnt/resource
else
    # EC2
    DATA_ROOT=/mnt
fi

if [ ! -d "$DATA_ROOT/$USER" ]; then
    sudo mkdir $DATA_ROOT/$USER
    sudo chown -R $USER. $DATA_ROOT/$USER

    mkdir $DATA_ROOT/$USER/data
    ln -s $DATA_ROOT/$USER/data $HOME/data
fi

# (Re)start the notebook process
cd $HOME/notebooks
pkill -9 -f "disabled-ipython-browser" || echo "Nothing to kill"
nohup ~/venv/bin/ipython notebook \
    --ip="*" \
    --browser="disabled-ipython-browser" &

