#!/bin/bash

python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config4.json -gpu 2
python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config5.json -gpu 2
python3 mains/mnist_multiview_main.py -c configs/mnist_multiview_config6.json -gpu 2
