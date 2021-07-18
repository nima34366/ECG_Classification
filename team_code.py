import numpy as np
from helper_code import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling1D, Conv1D, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from scipy.fft import rfft
import tensorflow.keras.backend as K
import os
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

def training_code(data_directory, model_directory):

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
    def generate_image(recording,label,frequency):
        recording = recording[:,:20*frequency]
        fftsignals = rfft(recording)
        fftsignals = fftsignals[:,:1024]
        mag = np.absolute(fftsignals)
        mag = np.clip(mag,0,500)
        angle = np.angle(fftsignals)
        image = np.vstack((mag,angle))
        image = np.transpose(image,(1,0)).astype(np.float32)
        label = label.astype(np.float32)
        image_byte=image.tobytes()
        label_byte=label.tobytes()
        dataset = {'image': _bytes_feature(image_byte), 'label': _bytes_feature(label_byte)}
        feature = tf.train.Features(feature=dataset)
        example = tf.train.Example(features=feature)
        serialized = example.SerializeToString()
        writer_train.write(serialized)

    labels=["164889003","164890007","6374002","426627000","733534002","713427006","270492004","713426002","39732003","445118002","251146004","698252002","426783006","284470004","10370003","365413008","427172004","164947007","111975006","164917005","47665007","427393009","426177001","427084000","164934002","59931005"]
    header_files,recording_files=find_challenge_files(data_directory)
    num_trainingdata_in_batch= len(header_files)//10
    for j in range(10):
        print('Preprocessing....'+str(j+1)+' out of 10')
        with tf.io.TFRecordWriter('processed_train'+str(j)+'.tfrecord') as writer_train:
            for i in range(j*num_trainingdata_in_batch,(j+1)*num_trainingdata_in_batch):
                header=load_header(header_files[i])
                recording=load_recording(recording_files[i])
                frequency=int(get_frequency(header))
                num_samples=int(get_num_samples(header))
                current_labels=get_labels(header)
                current_labels=["733534002" if j == "164909002" else "713427006" if j == "59118001" else "284470004" if j == "63593006" else "427172004" if j == "17338001" else j for j in current_labels]
                label=np.zeros(26,dtype=np.uint8)
                label_indices = [k for k in range(len(labels)) if labels[k] in current_labels]
                label[label_indices]=1
                recording=np.array(choose_leads(recording, header, twelve_leads),dtype=np.float32)
                adc_gains = get_adc_gains(header, twelve_leads).reshape(12, 1)
                baselines = get_baselines(header, twelve_leads).reshape(12, 1)
                recording = (recording - baselines)/adc_gains
                if num_samples<(22*frequency):
                    recording = np.hstack((recording,np.zeros((12,22*frequency-num_samples))))
                generate_image(recording, label, frequency)
    print('Preprocessing Done')

    def train_model(leads):
        
        def parse_example(serialized, image_shape=(1024, 24),label_shape=(26,)):
            features = {'image': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.string)}
            parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
            label_raw = parsed_example['label']
            image_raw = parsed_example['image']
            image = tf.io.decode_raw(image_raw, tf.float32)
            image = tf.reshape(image, shape=image_shape)
            image = tf.experimental.numpy.take(image,indices(leads),axis=-1)
            label = tf.io.decode_raw(label_raw, tf.float32)
            label = tf.reshape(label, shape=label_shape)
            return image,label

        AUTOTUNE=tf.data.AUTOTUNE
        train_files = tf.io.matching_files('processed_train*.tfrecord')
        train_files = tf.random.shuffle(train_files)
        shards = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = shards.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.map(parse_example, num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.batch(1024)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

        model = create_model(leads)

        model.fit(train_dataset,epochs=30)
        K.set_value(model.optimizer.learning_rate, 0.0001)
        model.fit(train_dataset,epochs=15)
        model.save_weights(model_directory+'/model'+str(len(leads))+'.hdf5')

    for leads in lead_sets:
        print('Training model'+str(len(leads)))
        train_model(leads)
    
    os.system('rm processed*')

def run_model(model, header, recording):

    def generate_image(recording,frequency,leads):
        recording = recording[:,:20*frequency]
        fftsignals = rfft(recording)
        fftsignals = fftsignals[:,:1024]
        mag = np.absolute(fftsignals)
        mag = np.clip(mag,0,500)
        angle = np.angle(fftsignals)
        image = np.vstack((mag,angle))
        image = np.transpose(image,(1,0)).astype(np.float32)
        image = np.take(image,indices_predicting(leads),axis=-1)
        return image
    classes=np.array(["164889003","164890007","6374002","426627000","733534002","713427006","270492004","713426002","39732003","445118002","251146004","698252002","426783006","284470004","10370003","365413008","427172004","164947007","111975006","164917005","47665007","427393009","426177001","427084000","164934002","59931005"])
    frequency=int(get_frequency(header))
    leads = get_leads(header)
    num_samples=int(get_num_samples(header))
    recording = np.array(choose_leads(recording, header, leads),dtype=np.float32)
    adc_gains = get_adc_gains(header, leads).reshape(len(leads), 1)
    baselines = get_baselines(header, leads).reshape(len(leads), 1)
    recording = (recording - baselines)/adc_gains
    if num_samples<(22*frequency):
        recording = np.hstack((recording,np.zeros((len(leads),22*frequency-num_samples))))
    image = np.array([generate_image(recording,frequency,leads)])
    probabilities = model.predict(image)
    probabilities = probabilities.reshape(26)
    labels = (probabilities>0.3)*1

    return classes, labels, probabilities

def load_model(model_directory, leads):

    model = create_model(leads)
    model.load_weights(model_directory+'/model'+str(len(leads))+'.hdf5')

    return model

def create_model(leads):
    
    model = Sequential()
    model.add(Conv1D(filters=72, groups=len(leads), kernel_size=7, activation='relu', input_shape=(1024,2*len(leads))))
    model.add(Conv1D(filters=72, groups=len(leads), kernel_size=7, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=144, groups=len(leads), kernel_size=9, activation='relu'))
    model.add(Conv1D(filters=144, groups=len(leads), kernel_size=9, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=288, groups=len(leads), kernel_size=11, activation='relu'))
    model.add(Conv1D(filters=288, groups=len(leads), kernel_size=11, activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26,activation = 'sigmoid', kernel_initializer='glorot_uniform'))
    AUROC = tf.keras.metrics.AUC(curve='ROC', name = 'AUROC',multi_label = True)
    AUPRC = tf.keras.metrics.AUC(curve='PR', name = 'AUPRC',multi_label = True)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', AUROC, AUPRC])
    model.summary()
    return model

def indices(leads):
    if leads==twelve_leads:
        pos = [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23]
    elif leads==six_leads:
        pos = [0,12,1,13,2,14,3,15,4,16,5,17]
    elif leads==four_leads:
        pos = [0,12,1,13,2,14,7,19]
    elif leads==three_leads:
        pos = [0,12,1,13,7,19]
    else:
        pos = [0,12,1,13]
    return pos

def indices_predicting(leads):
    if leads==twelve_leads:
        pos = [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23]
    elif leads==six_leads:
        pos = [0,6,1,7,2,8,3,9,4,10,5,11]
    elif leads==four_leads:
        pos = [0,4,1,5,2,6,3,7]
    elif leads==three_leads:
        pos = [0,3,1,4,2,5]
    else:
        pos = [0,2,1,3]
    return pos