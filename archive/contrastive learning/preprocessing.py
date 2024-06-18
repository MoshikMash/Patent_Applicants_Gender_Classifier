import json
import os
import pickle
import random

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import BertTokenizer


def create_preprocessed_data(config):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df = pd.read_csv(config['paths']['data_path'])
    df = df[['application_number', 'abstract']]
    df = df.set_index('application_number', drop=False)
    d = {}
    for application_number in list(df['application_number']):
        d[application_number] = tokenizer.encode_plus(df.loc[application_number]['abstract'], return_tensors='pt',
                                                      max_length=config['encoding_max_length'], truncation=True,
                                                      padding='max_length')

    with open(config['paths']['encoding_dictionary_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(d, file)


def create_pairs(config):
    df = pd.read_csv(config['paths']['data_path'])
    application_numbers = list(df['application_number'])
    labels = list(df['one_if_male'])
    num_pairs = config['split_into_train_validation_and_test']['number_of_pairs']
    balance_rate = config['split_into_train_validation_and_test']['balance_rate']
    # Partition application numbers based on labels
    same_label_numbers = [[] for _ in range(2)]
    for i, label in enumerate(labels):
        same_label_numbers[label].append((application_numbers[i], label))

    # Calculate the number of same and different label pairs based on balance rate
    num_same_label_pairs = int(num_pairs * balance_rate)
    num_different_label_pairs = num_pairs - num_same_label_pairs

    # Randomly select pairs from same label partition
    selected_same_label_pairs = []
    for _ in range(num_same_label_pairs):
        label = random.randint(0, 1)
        if len(same_label_numbers[label]) < 2:
            continue
        pair = random.sample(same_label_numbers[label], 2)
        selected_same_label_pairs.append(pair)

    # Randomly select pairs from different label partition
    selected_different_label_pairs = []
    for _ in range(num_different_label_pairs):
        label1, label2 = random.sample(range(2), 2)
        pair = [random.choice(same_label_numbers[label1]), random.choice(same_label_numbers[label2])]
        selected_different_label_pairs.append(pair)

    all_pairs = selected_same_label_pairs + selected_different_label_pairs
    random.shuffle(all_pairs)
    application_pairs = []
    same_label_flags = []
    for pair in all_pairs:
        application_pairs.append((pair[0][0], pair[1][0]))
        if pair[0][1] == pair[1][1]:
            same_label_flags.append(1)
        else:
            same_label_flags.append(0)
    return application_pairs, same_label_flags


def split_into_train_val_and_test(config):
    train_size = config['split_into_train_validation_and_test']['train_size']
    val_size = config['split_into_train_validation_and_test']['validation_size']
    test_size = 1 - train_size - val_size
    application_pairs, labels = create_pairs(config)
    # Split the data into train and test sets
    random_state = config['random_state']
    X_train_val, X_test, y_train_val, y_test = train_test_split(application_pairs, labels, test_size=test_size,
                                                                stratify=labels,
                                                                random_state=random_state)

    # Define StratifiedShuffleSplit for train-validation split
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=val_size, random_state=random_state)

    # Generate train and validation indices
    train_idx, val_idx = next(splitter.split(X_train_val, y_train_val))

    # Get train-validation data
    X_train, y_train = [X_train_val[i] for i in train_idx], [y_train_val[i] for i in train_idx]
    X_val, y_val = [X_train_val[i] for i in val_idx], [y_train_val[i] for i in val_idx]

    with open(config['paths']['X_train_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(X_train, file)
    with open(config['paths']['X_val_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(X_val, file)
    with open(config['paths']['X_test_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(X_test, file)
    with open(config['paths']['y_train_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(y_train, file)
    with open(config['paths']['y_val_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(y_val, file)
    with open(config['paths']['y_test_path'], 'wb') as file:
        # Serialize and save the object to the file
        pickle.dump(y_test, file)


def count_matching_labels(tuples_list):
    count = 0
    for tuple_pair in tuples_list:
        if tuple_pair[0][1] == tuple_pair[1][1]:
            count += 1
    return count


def preprocessing(config):
    if config['create_encoding_dictionary_flag']:
        random.seed(config['random_state'])
        os.makedirs('preprocessing', exist_ok=True)
        create_preprocessed_data(config)
    if config['split_into_train_validation_and_test']['split_into_train_validation_and_test_flag']:
        os.makedirs('preprocessing', exist_ok=True)
        split_into_train_val_and_test(config)



