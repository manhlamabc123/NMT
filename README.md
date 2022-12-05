# Neural Machine Translation on multiple Model

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#technologies">Technologies</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#how-to-run">How to run</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About

* The actually first project AI that I created from scratch (From Data Processing to Create Model, Train & Evalute, ...)
* This project is created to test multiple Model Seq2Seq on the same dataset
* Statuse: **In Progress**. The project is ready, just the code base right now is very messy.

## Architecture

* Now avaiable:
  * Encoder (Bi-GRU) -> Decoder (GRU)
  * Encoder (Bi-GRU) -> Attention -> Decoder (GRU)
  * Transformer

## Technologies

* Language: Python 3.9.12
* Framework: PyTorch 1.9.0+cu102
* Required Libraries:
  * numpy 1.23.1
  * nltk 3.7
  * matplotlib 3.5.2
  * pandas 1.5.1
  * underthesea 1.3.5
  
## Dataset
* Source: [Kaggle](https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation)
* Already downloaded in `data` folder

## How to run

### Run this for details
```
python main.py -h
```

### Data preparation
```
python main.py -d
```

### To train the model
```
python main.py -t -m model_name
```
Please replace `model_name` with the following:
* `gru`: Encoder (Bi-GRU) -> Decoder (GRU)
* `attention`: Encoder (Bi-GRU) -> Attention -> Decoder (GRU)
* `transformer`: Transformer
`-m model_name` is not required, if blank, default to `gru`

### To validate the model
```
python main.py -e -m model_name
```
Replace `model_name` like above.

### To plot attention
```
python main.py -a
```
This only works with `attention` model

## Acknowledgments

* [This tutorial on Seq2Seq with Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [This tutorial on creating a Chatbot using Seq2Seq](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
* [This tutorial on training with PyTorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
* [This blog on NMT](https://medium.com/@rishikesh_d/neural-machine-translation-a-comprehensive-guide-ef414e79b49)
* [Code for Transformer](https://www.kaggle.com/code/huhuyngun/english-to-vietnamese-with-transformer)
* [Code for Attention mechanism](https://www.kaggle.com/code/minhhngchong/machine-translation-en-to-vi-with-attention)
