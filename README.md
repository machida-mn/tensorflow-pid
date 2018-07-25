# PID optimizer for Tensorflow

Tensorflow implementation of PID optimization (CVPR 2018).

See the following paper for detail of algorithm.

- [A PID Controller Approach for Stochastic Optimization of Deep Networks](http://www4.comp.polyu.edu.hk/%7Ecslzhang/paper/CVPR18_PID.pdf)

This program was tested on Tensorflow r1.4.

## Installation

```console
$ git clone https://github.com/machida-mn/tensorflow-pid
$ cd tensorflow-pid
$ pip install .
```

## Usage

```python
import tensorflow_pid

loss = .......

train_op = tensorflow_pid.PIDOptimizer(learning_rate=0.01, kd=0.001).minimize(loss)
```

## Implementation in other frameworks

- PyTorch
    - [PIDOptimizer (GitHub)](https://github.com/tensorboy/PIDOptimizer)
