import tqdm
import deepsmiles
import numpy as np
import tensorflow as tf
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from . import I2S_Data


CONVERTER = deepsmiles.Converter(rings=True, branches=True)


def deep2smi(smi):
    try:
        return CONVERTER.decode(smi)
    except deepsmiles.DecodeError:
        return ''


class Trainer:
    def __init__(self, encoder, decoder, optimizer, tokenizer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    
    @tf.function
    def train_step(self, img_tensor, target, loss_function):
        ## initializing the hidden state for each batch because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        loss = 0
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            
            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += loss_function(target[:, i], predictions)
                dec_input = tf.expand_dims(target[:, i], 1)
        
            total_loss = (loss / int(target.shape[1]))
        
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    
        return loss, total_loss
    
    
    def load_checkpoint(self, checkpoint_path):    
        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    

    def evaluate(self, all_img_name, image_features_extract_model, maxlength=74, use_tqdm=True):
        predicted = []
        if use_tqdm: pbar = tqdm.tqdm(all_img_name)
        else: pbar = all_img_name
        for filepath_img in pbar:
            res = self.eval_step(filepath_img, image_features_extract_model, maxlength)
            smi = ''.join(res).replace("<start>","").replace("<end>","").replace("<unk>","")
            if len(smi) > 0:
                smi = deep2smi(smi)
            predicted.append(smi)
        
        return predicted

    
    def eval_step(self, image, image_features_extract_model, maxlength):    
        ## input data    
        temp_input = tf.expand_dims(I2S_Data.load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        ## encoding
        features = self.encoder(img_tensor_val)

        ## decoding
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        hidden = self.decoder.reset_state(batch_size=1)
        
        result = []
        for i in range(maxlength):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
    
            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(self.tokenizer.index_word[predicted_id])
    
            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result
    
            dec_input = tf.expand_dims([predicted_id], 0)
    
        return result


def calc_tanimoto_similarity(x, y):
    score = 0.
    try:
        mol1 = Chem.MolFromSmiles(x)
        mol2 = Chem.MolFromSmiles(y)        
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        score += DataStructs.FingerprintSimilarity(fp1, fp2)
    except:
        pass
    return score


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


def create_dataset(img_name_train, cap_train, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ## shuffling and batching
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset