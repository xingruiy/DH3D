#!/bin/bash

cd /tmp
wget https://github.com/NVlabs/cub/archive/v1.8.0.zip
unzip v1.8.0.zip -d $HOME/libs
export CUB_INC=$HOME/libs/cub-1.8.0/
rm /tmp/v1.8.0.zip