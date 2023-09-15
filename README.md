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

## Training results

```
[I 2023-09-15 13:04:12,225] A new study created in memory with name: no-name-bfc99bd8-2b90-4913-a979-64fcfe736f77
[I 2023-09-15 13:33:39,951] Trial 0 finished with value: 0.1135 and parameters: {'batch_size': 58, 'lr': 0.007533987858883899}. Best 
is trial 0 with value: 0.1135.
[I 2023-09-15 13:52:49,840] Trial 1 finished with value: 0.1032 and parameters: {'batch_size': 126, 'lr': 0.06902812409998378}. Best 
is trial 0 with value: 0.1135.
[I 2023-09-15 14:25:05,687] Trial 2 finished with value: 0.1028 and parameters: {'batch_size': 54, 'lr': 0.027973357381960676}. Best 
is trial 0 with value: 0.1135.
[I 2023-09-15 14:45:47,737] Trial 3 finished with value: 0.1135 and parameters: {'batch_size': 111, 'lr': 0.022178308944666807}. Best is trial 0 with value: 0.1135.
[I 2023-09-15 15:05:36,712] Trial 4 finished with value: 0.1028 and parameters: {'batch_size': 126, 'lr': 0.06211682342661622}. Best 
is trial 0 with value: 0.1135.
Number of finished trials: 5
Best trial:
  Value: 0.1135
  Params: {'batch_size': 58, 'lr': 0.007533987858883899}
  ```