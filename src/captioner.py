import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re # REEEEEEEEEEE
import numpy as np
import os
import sys
import time
import json
from glob import glob
from PIL import Image
import pickle

# import my models.py
from models import *
from training import *

def main():
    # Big main O_O

    # get annotation filepath
    annotation_folder = '/data/training/palettes/'
    if not os.path.exists(os.path.abspath('..') + annotation_folder):
        print("annotation directory", annotation_folder, " not found")
        sys.exit()

    annotation_file = os.path.dirname(os.path.abspath('..') + annotation_folder) + '/annotations.json'

    # get image filepath
    image_folder = '/../data/training/bgs/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        print("image directory", annotation_folder, " not found")
        sys.exit()

    PATH = os.path.abspath('.') + image_folder

    # read json
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []
    for annot in annotations['annotations']:
        caption = '<start>' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_path = PATH + (image_id)

        all_img_name_vector.append(full_path)
        all_captions.append(caption)

    # shuffle captions and image_names together
    # set a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                            all_img_name_vector,
                                            random_state=1)

    # use imagenet weights for convenience
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # get the first 5000 sequences
    top_k = 5000

    # make tokens from sequences, filter
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                        oov_token="<unk>",
                                                        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # pad sequences to longest len
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_seqs = tokenizer.texts_to_sequences(train_captions)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    max_length = calc_max_length(train_seqs)

    #split data to training/testing 80/20
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)
    
    # print the number of entries for the sets
    len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

    # Create dataset
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1
    num_steps = len(img_name_train)

    #shape of vector from inceptionV3
    features_shape = 2046
    attention_features_shape = 64

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # use map to load numpy files
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                map_func, [item1, item2], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


    # shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction='none')

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                                decoder=decoder,
                                optimizer = optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if skpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # training
    loss_plot = []
    
    EPOCHS = 20
    for epoch in range(start_epoch,EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
         
        if epoch % 5 == 0:
            ckpt_manager.save()

        # Print some training info
        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss/num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        plt.plot(loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.show()



def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.numpy')
    return img_tensor, cap


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


if __name__ == "__main__":
    main()
