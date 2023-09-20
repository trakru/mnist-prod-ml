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

Available on the local Optuna dashboard

run `optuna-dashboard sqlite:///db.sqlite3` to view training results

### Note on CUDA with Package Managers (Pip/Conda)

For local installs only

Cuda tends to not play nice with `pip` based installs. The recommended resolution is to create conda environments by running `conda create --name env` and running `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`. ALternatively, if we still would want to use pip, use the following to get torch installed `pip install --upgrade --force-reinstall torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117`

Cloud-based workflows are not impacted. just use one of the pre-configured images with torch & cuda-enabled
