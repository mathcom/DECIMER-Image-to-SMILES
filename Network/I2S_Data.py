import os
import re
import sys
import time
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def split_by_vocabulary(text, vocabulary):
    # 정규표현식으로 vocabulary 단어를 OR로 연결
    vocab_pattern = '|'.join(re.escape(word) for word in vocabulary)
    
    # 주어진 패턴에 맞게 문자열을 split
    split_pattern = f'({vocab_pattern})'
    
    # split 결과에서 빈 문자열 제거
    tokens = [token for token in re.split(split_pattern, text) if token]
    
    return tokens


## Create Inception V3 Model
def get_image_features_extract_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


## Creating tokenizer with defined characters
def create_tokenizer(tests, max_voc = 100):    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_voc, oov_token="<unk>", filters='!"$&:;?^`{}~ ', lower=False)
    tokenizer.fit_on_texts(tests)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    return tokenizer


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def data_loader(filepath, image_dir, train_test_split=True):
    with open(filepath, 'r') as txt_file:
        '''
        Image_id,DeepSMILES
        CDK_Depict_40_9,CCC=C=CCC)PC63C
        CDK_Depict_5_28,C=NC=CN=CPC5=N9
        CDK_Depict_47_200,C=CCN=CN)SN
        '''
        smiles = txt_file.read()

    all_smiles = []
    all_img_name = []
    
    ## Splitting the text file and saving SMILES as Captions
    for line in smiles.split('\n')[1:]: # skip header
        if len(line) > 0:
            tokens = line.split(',')
            image_id = str(tokens[0])+'.png'
            
            ## label
            try:
                caption = '<start>' + str(tokens[1].rstrip()) + '<end>'
            except IndexError as e:
                print (e, flush=True)
            
            all_smiles.append(caption)
            
            ## image filepath
            full_image_path = os.path.join(image_dir, image_id)
            all_img_name.append(full_image_path)
            

    print("Selected Data ",len(all_smiles), "All data ", len(all_smiles))
    
    for x, y in zip(all_smiles[:5], all_img_name[:5]):
        print(x, y)

    ## Loading InceptionsV3 Model to convert Images to Numpy arrays (features)
    image_features_extract_model = get_image_features_extract_model()

    encode_train = sorted(set(all_img_name))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(100)

    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            try:
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())
            except OSError as e:
                print (e)

    if train_test_split:
        ## Splitting the dataset to train and test, in our case the test set is 10% of the train set. And completely unseen by the training process.
        img_name_train, img_name_val, smi_train, smi_val = train_test_split(all_img_name, all_smiles, test_size=0.1, random_state=0, shuffle=False)
    else:
        img_name_train = all_img_name
        img_name_val = None
        smi_train = all_smiles
        smi_val = None

    return img_name_train, img_name_val, smi_train, smi_val, image_features_extract_model


def data_loader_eval(filepath, image_dir):
    with open(filepath, 'r') as txt_file:
        '''
        Image_id,DeepSMILES
        CDK_Depict_40_9,CCC=C=CCC)PC63C
        CDK_Depict_5_28,C=NC=CN=CPC5=N9
        CDK_Depict_47_200,C=CCN=CN)SN
        '''
        smiles = txt_file.read()

    all_smiles = []
    all_img_name = []
    
    ## Splitting the text file and saving SMILES as Captions
    for line in smiles.split('\n')[1:]: # skip header
        if len(line) > 0:
            tokens = line.split(',')
            image_id = str(tokens[0])+'.png'
            try:
                caption = str(tokens[1].rstrip())
            except IndexError as e:
                print (e,flush=True)
            full_image_path = os.path.join(image_dir, image_id)

            all_img_name.append(full_image_path)
            all_smiles.append(caption)
            
    return all_smiles, all_img_name