# MNIST Digits Classification

This project offers a solution for classifying MNIST handwritten digits using a Convolutional Neural Network (CNN) and PyTorch according to the specifications in the [TASK document](TASK.md)

## Network Architecture

The neural network consists of the following layers:

| Layer Name   | Type           | Description               | Dimension     |
|--------------|----------------|---------------------------|---------------|
| conv1        | Convolutional  | First layer               | 28x28x32      |
| conv2_1      | Convolutional  | Parallel layer            | 14x14x64      |
| conv2_2      | Convolutional  | Parallel layer            | 14x14x64      |
| conv3_1      | Convolutional  | Sequential layer          | 7x7x512       |
| conv3_2      | Convolutional  | Sequential layer          | 7x7x512       |
| fc1          | Fully Connected| -                         | 1000          |
| fc2          | Fully Connected| -                         | 500           |
| output_layer | Output         | -                         | 10         