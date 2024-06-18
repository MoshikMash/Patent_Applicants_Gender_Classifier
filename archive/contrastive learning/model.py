import os
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel
import torch.nn.functional as F

from dataset import PatentsDataset


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.batch_size = config['batch_size']

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.device)
        input_size = self.bert_model.config.hidden_size

        self.pooler_type = config['pooler_type']

        if self.pooler_type == 'lstm':
            self.pooler = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=1024 // 2,
                                        bidirectional=True)
            self.pooler.flatten_parameters()
        elif self.pooler_type == 'NN':
            self.pooler = torch.nn.Sequential(
                torch.nn.Linear(input_size, 512),  # Input size to the first hidden layer
                torch.nn.ReLU(),  # Activation function
                torch.nn.Linear(512, 256),  # First hidden layer to the second hidden layer
                torch.nn.ReLU()
                # torch.nn.Linear(256, 128),  # Second hidden layer to the output layer
                # torch.nn.ReLU(),

            )
        # self.pooler = CustomPooler(input_size)
        self.pooler.to(self.device)

        with open(config['paths']['encoding_dictionary_path'], 'rb') as file:
            self.encodings = pickle.load(file)

        self.freeze_bert_layers = config['freeze_bert_layers_flag']
        if self.freeze_bert_layers:
            # Freeze the parameters of the BERT model
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, eps=1e-8)

    def forward(self, inputs):
        outputs_a, outputs_b = [], []
        for i in range(len(inputs[0])):
            input_data_a = self.encodings[inputs[0][i].item()].to(self.device)
            output_a = self.bert_model(input_data_a['input_ids'], input_data_a['attention_mask']).last_hidden_state[:,
                       -1, :]
            output_a = self.pooler(output_a)[0]
            outputs_a.append(output_a)

            input_data_b = self.encodings[inputs[1][i].item()].to(self.device)
            output_b = self.bert_model(input_data_b['input_ids'], input_data_b['attention_mask']).last_hidden_state[:,
                       -1, :]
            output_b = self.pooler(output_b)[0]
            outputs_b.append(output_b)

        return outputs_a, outputs_b

    def cosine_distance_loss(self, outputs_a, outputs_b, targets, config):
        device = config['device']
        outputs_a_tensor = torch.stack([embedding.clone().detach() for embedding in outputs_a]).to(device)
        outputs_b_tensor = torch.stack([embedding.clone().detach() for embedding in outputs_b]).to(device)

        outputs_a_normalized = F.normalize(outputs_a_tensor, p=2, dim=-1)
        outputs_b_normalized = F.normalize(outputs_b_tensor, p=2, dim=-1)

        cosine_similarities = torch.sum(outputs_a_normalized * outputs_b_normalized, dim=-1)

        loss = 1 - cosine_similarities

        if targets is not None:
            if self.pooler_type == 'lstm':
                targets_tensor = targets.clone().detach().unsqueeze(1).to(device)  # Convert to column tensor
                loss += targets_tensor
            elif self.pooler_type == 'NN':
                targets = targets.to(self.device)
                loss += targets

        loss = torch.mean(loss).requires_grad_(True)

        targets = targets.tolist()
        # Calculate predicted probabilities
        predicted_probs = cosine_similarities.flatten().tolist()

        return loss, predicted_probs, targets


def train(config):
    # Load the object from the file using pickle
    with open(config['paths']['X_train_path'], 'rb') as file:
        X_train = pickle.load(file)
    with open(config['paths']['y_train_path'], 'rb') as file:
        y_train = pickle.load(file)

    with open(config['paths']['X_val_path'], 'rb') as file:
        X_val = pickle.load(file)
    with open(config['paths']['y_val_path'], 'rb') as file:
        y_val = pickle.load(file)
    model = Model(config)
    dataset_train = PatentsDataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=model.batch_size, shuffle=False)
    dataset_val = PatentsDataset(X_val, y_val)
    dataloader_val = DataLoader(dataset_val, batch_size=model.batch_size, shuffle=False)

    # Training loop
    save_model_interval = config['save_model_interval']  # Save every 100 iterations as default
    model_save_path = config['paths']['model_save_path']  # Default path to save models
    os.makedirs(model_save_path, exist_ok=True)
    previous_model_path = None
    last_model_path = None

    train_interval = config['train_interval']
    validation_interval = config['validation_interval']

    iteration = 0
    model.train()  # Set the model to train mode
    for inputs, targets in dataloader_train:
        iteration += 1
        model.optimizer.zero_grad()

        outputs_a, outputs_b = model(inputs)
        loss, predicted_probs, targets = model.cosine_distance_loss(outputs_a, outputs_b, targets, config)

        loss.backward()
        if model.freeze_bert_layers:
            torch.nn.utils.clip_grad_norm(model.pooler.parameters(), 1.0)
        else:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        model.optimizer.step()

        # print train's batch accuracy
        if iteration % train_interval == 0:
            print(f'Train Loss: {loss}')

        # Perform validation
        if iteration % validation_interval == 0:
            validation_loss = 0.0
            validation_scores = []
            validation_samples = 0
            with torch.no_grad():
                for inputs_val, targets_val in dataloader_val:
                    outputs_a_val, outputs_b_val = model(inputs_val)
                    loss_val, predicted_probs_val, targets_val = model.cosine_distance_loss(outputs_a_val,
                                                                                            outputs_b_val,
                                                                                            targets_val, config)
                    auc_score_val = roc_auc_score(targets, predicted_probs)

                    validation_loss += loss_val.item()
                    validation_scores.append(auc_score_val)
                    validation_samples += len(targets_val)

            print('-----')
            print(
                f"Validation Loss: {validation_loss / len(dataloader_val)}, "
                f"Validation {config['metric']}: {np.mean(validation_scores)}"
            )
            print('-----')

        # Save model checkpoint
        if iteration % save_model_interval == 0:
            if previous_model_path is not None and os.path.exists(previous_model_path):
                os.remove(previous_model_path)

            # Update the path for the model to be considered 'previous' in the next iteration
            previous_model_path = last_model_path

            # Save the new model checkpoint
            checkpoint_path = os.path.join(model_save_path, f'model_checkpoint_{iteration}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved to {checkpoint_path} at iteration {iteration}')

            # Update last_model_path to the current checkpoint
            last_model_path = checkpoint_path
