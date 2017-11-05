#!/bin/bash

# Plots logged GPU memory usage.
# Use output of log_mem_stats.sh as the $1 parameter.

echo 'x=[' >mem_nums.py; grep cifar10_train $1 | sed -e "s/^.* \([0-9]\+\)MiB .*$/\1,/" >>mem_nums.py; echo ']' >>mem_nums.py

python3 <<EOF
import matplotlib.pyplot as plt
exec(open('mem_nums.py').read())
plt.plot(x)
plt.show()
EOF

rm mem_nums.py
