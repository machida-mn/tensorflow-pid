#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))

import tensorflow_pid  # noqa:E402


def nearly_equal(x: float, y: float, max_error: float = 1e-6) -> bool:
    return abs(x - y) < max_error


def test_minimize_square() -> None:
    epoch = 100

    x = tf.get_variable('x', initializer=1.0)
    y = tf.square(x)
    train = tensorflow_pid.PIDOptimizer(learning_rate=0.1, kd=0.01).minimize(y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            sess.run(train)

        x_val = sess.run(x)
        assert nearly_equal(x_val, 0.)


if __name__ == '__main__':
    test_square_minimize()
