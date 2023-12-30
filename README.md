# bertorso
ai
terminal

https://www.tensorflow.org/text/tutorials/classify_text_with_bert 
pip install -U "tensorflow-text==2.13.*"
pip install "tf-models-official==2.13.*"



Code

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

