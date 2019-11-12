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

