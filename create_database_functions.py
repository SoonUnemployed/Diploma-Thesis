import os
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch

#Keep only npz files
def get_files(file_path):
    files =[f for f in os.listdir(file_path) if f.endswith(".npz")]
    all_data = {}

    for filename in files:
        data = np.load(os.path.join(file_path, filename))
        filename = filename[:-4] #removes npz
        
        features = data["features"]
        user_id = data["user_id"]
        stimuli_sequence = data["stimuli_sequence"]
        
        all_data[filename] = {"features": features, 
                            "user_id": int(user_id), 
                            "stimuli_sequence": stimuli_sequence,
                            "extraction_model": int(filename[-1:]) #Feature Extraction Model
                            }
    return all_data

#We need to compute the averages to use later for the cosine similarity model 
def get_averages(all_data):
    #Set so only unique values stay
    user_ids = set(info["user_id"] for info in all_data.values())
    av_per_user = {}

    for user_id in user_ids:
        #Feutures for each user in list 
        user_features_1 = [
                            np.array(info["features"])
                            for _, info in all_data.items() #_, because we don't care about the filename
                            if info["user_id"] == user_id and info["extraction_model"] == 1
                        ]
        
        stim_sequece_1 = [
                            info["stimuli_sequence"]
                            for _, info in all_data.items()
                            if info["user_id"] == user_id and info["extraction_model"] == 1
                        ]
        
        user_features_2 = [
                            np.array(info["features"])
                            for _, info in all_data.items() 
                            if info["user_id"] == user_id and info["extraction_model"] == 2
                        ]
        
        stim_sequece_2 = [
                            info["stimuli_sequence"]
                            for _, info in all_data.items() 
                            if info["user_id"] == user_id and info["extraction_model"] == 2
                        ]
        
        #Average per session for each user per model
        session_avgs_1 = [np.mean(feature, axis = 0) for feature in user_features_1]
        session_avgs_2 = [np.mean(feature, axis = 0) for feature in user_features_2]
        
        #Get the averages per user for each model
        av_per_user[(user_id, 1)] = np.mean(session_avgs_1, axis = 0)
        av_per_user[(user_id, 2)] = np.mean(session_avgs_2, axis = 0)

        #Get each embedding array from the user_features
        each_freq_av = {}
            
        for cl_num in set(stim_sequece_1[0]):
            temp_freq_list = []
            for i in range(len(user_features_1)):
                for j in range(len(stim_sequece_1[i])):
                    if stim_sequece_1[i][j] == cl_num:
                        temp_freq_list.append(user_features_1[i][j])
            each_freq_av[(user_id, cl_num, 1)] = np.mean(temp_freq_list, axis = 0)

        for cl_num in set(stim_sequece_2[0]):
            temp_freq_list = []
            for i in range(len(user_features_2)):
                for j in range(len(stim_sequece_2[i])):
                    if stim_sequece_2[i][j] == cl_num:
                        temp_freq_list.append(user_features_2[i][j])
            each_freq_av[(user_id, cl_num, 2)] = np.mean(temp_freq_list, axis = 0)


        return av_per_user, each_freq_av
    
#Classification Layer for the data without the labels
class UserClassifier(nn.Module):
    def __init__(self, input_channels, time_length, num_classes = 12):
        super(UserClassifier, self).__init__()

        self.classifier_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input_channels,
                out_channels = 12,
                kernel_size = (1, 8),
                stride = (1, 1),
                padding = 'same',
                bias = False
            ),
            nn.BatchNorm2d(12, eps = 1e-5, momentum = 0.01, affine = True, track_running_stats = True),
            nn.ELU(alpha = 1.0),
            nn.Dropout(p = 0.5, inplace = False),
            nn.AvgPool2d(kernel_size = (1, 2), stride = (1, 2), padding = 0)
        )
        #Use dummy data to calculate the Linear size frot he 2 extractors
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 1, time_length)
            dummy_out = self.classifier_1(dummy)
            self.flattened_size = dummy_out.view(1, -1).size(1)

        self.classifier_2 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(self.flattened_size, num_classes)
        )

    def forward(self, x):
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        return x
    
#NN with classification layer for features with stimuli label 
class EEGUserClassifier(nn.Module):
    def __init__(self, input_channels, time_length, num_classes = 12):
        super(EEGUserClassifier, self).__init__()

        self.classifier_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input_channels,
                out_channels = 12,
                kernel_size = (1, 8),
                stride = (1, 1),
                padding = 'same',
                bias = False
            ),
            nn.BatchNorm2d(12, eps = 1e-5, momentum = 0.01, affine = True, track_running_stats = True),
            nn.ELU(alpha = 1.0),
            nn.Dropout(p = 0.5, inplace = False),
            nn.AvgPool2d(kernel_size = (1, 2), stride = (1, 2), padding = 0),
            nn.Conv2d(
                in_channels = 12,
                out_channels = 6,
                kernel_size = (1, 4),
                stride = (1, 1),
                padding = 'same',
                bias = False
            ),
            nn.BatchNorm2d(6, eps = 1e-5, momentum = 0.01, affine = True, track_running_stats = True),
            nn.ELU(alpha = 1.0),
            nn.Dropout(p = 0.5, inplace = False),
            nn.AvgPool2d(kernel_size = (1, 2), stride = (1, 2), padding = 0)
        )
        #Use dummy data to calculate the Linear size frot he 2 extractors
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 1, time_length)
            dummy_out = self.classifier_1(dummy)
            self.flattened_size = dummy_out.view(1, -1).size(1)

        self.classifier_2 = nn.Sequential(
            nn.Flatten(start_dim = 1, end_dim = -1),
            nn.Linear(self.flattened_size, num_classes)
        )

    def forward(self, x):
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        return x
    
#Training model for both Classifiers 
def train_classifier(model, train_loader, n_epochs = 10, lr = 1e-3, device = "cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() #loss for classification
    optimizer = optim.Adam(model.parameters(), lr = lr) #learning rate(weight updates)

    for epoch in range(n_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim = 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{n_epochs} | Loss={total_loss:.4f} | Acc={acc:.4f}")

    return model

#Get the features and the user_ids (plus the stimuli sequence to use later)
def make_lists(all_data, extraction_model):
    features = []
    stimuli = []
    user_ids = []
    for info in all_data.values(): 
        if info["extraction_model"] == extraction_model:
            features.append(np.array(info["features"]))
            stimuli.append(np.array(info["stimuli_sequence"]))
            user_ids.append(info["user_id"])
    return features, stimuli, user_ids

#Only features Datasets
def make_databases(features_1, features_2, user_ids_1, user_ids_2):
    x_1 = torch.tensor(np.vstack(features_1), dtype = torch.float32) 
    y_1 = np.repeat(user_ids_1, 15)
    y_1 = torch.tensor(np.hstack(y_1), dtype = torch.long)

    x_2 = torch.tensor(np.vstack(features_2), dtype = torch.float32)
    y_2 = np.repeat(user_ids_2, 15)
    y_2 = torch.tensor(np.hstack(y_2), dtype = torch.long)
    return x_1, x_2, y_1, y_2