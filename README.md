# bertorso
ai
terminal
------------------------------------------------------------
https://demos.explosion.ai/displacy

-------------------------------------------------------
https://schreiber-ehle.de/mantisbt/view.php?id=40

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Tesla is looking a buzing U.S. startup for $6 million')
for token in doc:
    print(token.text, token.pos_, token.dep_)



--------------------------------------------------------------------------------------\
https://stackoverflow.com/questions/28618400/how-to-identify-the-subject-of-a-sentence
pip install rake_nltk
from rake_nltk import Rake

rake = Rake()

kw = rake.extract_keywords_from_text("Can Python + NLTK be used to identify the subject of a sentence?")

ranked_phrases = rake.get_ranked_phrases()

print(ranked_phrases)
-------------------------------------------------
python -m spacy download en_core_web_lg

import spacy 
nlp = spacy.load('en_core_web_lg')


def get_subject_object_phrase(doc, dep):
    doc = nlp(doc)
    for token in doc:
        if dep in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
    return str(doc[start:end])



-------------------------------------------------------------------------
https://stackoverflow.com/questions/28618400/how-to-identify-the-subject-of-a-sentence \
pip instal spacy
python -m spacy download en_core_web_sm 

import spacy
nlp = spacy.load('en')
sent = "I shot an elephant"
doc=nlp(sent)

sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]

print(sub_toks) 
------------------------------------------------------------------------------
http://jalammar.github.io/illustrated-bert/


-----------------------------------------------------------------------
https://www.mzes.uni-mannheim.de/socialsciencedatalab/article/bert-explainable-ai/ \

pip install transformers
python -m pip install --upgrade pip

from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sample = ("Mannheim is a beautiful city. It's close to two rivers and quite green.").lower()
encoding = tokenizer.encode(sample)
print(tokenizer.convert_ids_to_tokens(encoding))

print(encoding)

------------------------------------------------------------------------

https://www.tensorflow.org/text/tutorials/classify_text_with_bert \
pip install -U "tensorflow-text==2.13.*" \
pip install "tf-models-official==2.13.*" \



Code \
-----------code 1 ----------------------------
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

// Download Data

url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

# remove unused folders to make it easier to load the data
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
-----------------------------------new code 

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

--------------------------------------------------------------------
