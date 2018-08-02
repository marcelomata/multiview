#!/bin/bash

python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config2.json -gpu 1
python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config3.json -gpu 1
