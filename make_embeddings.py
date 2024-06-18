import json
import pickle

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Define function to tokenize text and get BERT embeddings
def get_bert_embeddings(config, text, tokenizer, bert_model,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    input_ids = tokenizer(text, return_tensors=config['tokenizer_params']['return_tensors'],
                          padding=config['tokenizer_params']['padding'],
                          truncation=config['tokenizer_params']['truncation'],
                          max_length=config['tokenizer_params']['max_length'])["input_ids"]

    # Move input tensor to the appropriate device (CPU or GPU)
    input_ids = input_ids.to(device)

    # Ensure the input tensor doesn't require gradients
    with torch.no_grad():
        # Move the model to the appropriate device
        bert_model = bert_model.to(device)

        # Perform inference on the model
        outputs = bert_model(input_ids)

    # Extract the last hidden state
    last_hidden_states = outputs.last_hidden_state

    # Move the tensor to CPU if necessary
    if device == torch.device("cuda"):
        last_hidden_states = last_hidden_states.cpu()

    # Convert to NumPy array
    return last_hidden_states[:, 0, :].numpy()


def make_embeddings(config):
    df = pd.read_csv(config['paths']['date_path'])

    data = pd.DataFrame({
        'abstract': df['clean_abstract'].values,
        'label': df['one_if_male'].values
    })

    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        deberta_model = AutoModel.from_pretrained(config['base_model'])
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the appropriate device
    deberta_model.to(device)

    print("Model and tokenizer loaded successfully.")

    # Tokenize abstracts and get BERT embeddings
    embeddings = []
    for abstract in tqdm(data['abstract']):
        embeddings.append(get_bert_embeddings(config, abstract, tokenizer, deberta_model, device))
    embeddings = np.concatenate(embeddings, axis=0)

    # Use 'wb' to write in binary mode
    with open(config['paths']['embeddings_path'], 'wb') as file:
        pickle.dump(embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("embeddings has been pickled and saved to", config['paths']['embeddings_path'])
