#!/bin/bash

python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config0.json -gpu 0
python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config1.json -gpu 0
