# GAN
Using a Generative Adversarial Network to Generate Images

## Neural Network Model

A GAN consists of two Neural Networks a Generative Network and a Discriminative Networks.

- **Generative Netork**: It is a network which takes in a random noise vector and generates images that are **"similar"** to the input images.
<p align="center">
<img src="https://github.com/crypto-code/GAN/blob/master/assets/gen_model.png" width="800" align="middle" />   </p>


- **Adversarial Network**: It is a network which takes in **"Fake"** and Real images, and acts as a binary classifier, classifying whether the input is fake or real.  
<p align="center">
<img src="https://github.com/crypto-code/GAN/blob/master/assets/disc_model.jpg" width="800" align="middle" />   </p>


Together the networks will try to decrease their respective losses, hence the name **"Adverserial Netowrks"**
<p align="center">
<img src="https://github.com/crypto-code/GAN/blob/master/assets/model.png" width="800" align="middle" />   </p>

## Requirements:
* Python 3.6.2 (https://www.python.org/downloads/release/python-362/)
* Numpy (https://pypi.org/project/numpy/)
* Tensorflow (https://pypi.org/project/tensorflow/)
* Keras (https://pypi.org/project/Keras/)
* OpenCV (https://pypi.org/project/opencv-python/)

## Usage:
- Download the required images, and put it in the data folder:

**Note: To get the best result ensure that the images don't vary dratically**

- To preprocess the images run resize.py, followed by RGBA2RGB.py with the following argumanets
```
usage: resize.py [-h] --input INPUT --output OUTPUT

Resize Input Images

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Directory containing images to resize. eg: ./data
  --output OUTPUT  Directory to save resized images. eg: ./resized
```

```
usage: RGBA2RGB.py [-h] --input INPUT --output OUTPUT

Convert RGBA to RGB

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Directory containing images to resize. eg: ./resized
  --output OUTPUT  Directory to save resized images. eg: ./RGB_data
```

- Now you can run the GAN on the final processed images, using GAN.py
```
usage: GAN.py [-h] --mode MODE [--name NAME] [--input INPUT] [--output OUTPUT]
              [--epoch EPOCH] [--batch BATCH]

Train or Test the Generative Adverserail Network

optional arguments:
  -h, --help       show this help message and exit
  --mode MODE      Whether to Test or Train
  --name NAME      Directory of the Generated Images eg: NewPaints
  --input INPUT    Directory of input Images eg: RGB_data
  --output OUTPUT  Output Image Name
  --epoch EPOCH    Number of Epochs to Run
  --batch BATCH    Batch Size
```

## Generated Examples

The GAN was trained for 29 hrs on a dataset consisting of various abstract paintings, and it came up with these

<p align="center">
<img src="https://github.com/crypto-code/GAN/blob/master/assets/example.jpg" align="middle" />   </p>

These aren't perfect because perfecting the generator requires a lot of computational power.

# G00D LUCK

For doubts email me at:
atinsaki@gmail.com
