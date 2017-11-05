#!/bin/bash

# keeps writing NVidia's GPU stats to a file and the console.

while true; do nvidia-smi; sleep 1; done | tee nvidia-smi.log
