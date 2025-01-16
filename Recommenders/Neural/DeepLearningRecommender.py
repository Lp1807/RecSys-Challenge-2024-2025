from Recommenders.BaseRecommender import BaseRecommender
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeepLearningRecommender(nn.Module, BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        super().__init__()
        BaseRecommender.__init__(self, URM_train, verbose)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def _data_generator(self, batch_size, num_negatives=3, num_items=None):
        user_input, item_input, labels = [], [], []
        dok_train = URM_train.todok() # <- Dictionary representation of a sparse matrix: allows us to check existing interactions as key-value pairs
        if num_items is None : num_items = self.URM_train.shape[1]

        self.batch_counter = 0
        start = self.batch_counter
        stop = min(self.batch_counter + batch_size, len(dok_train.keys()))
        for (u,i) in dok_train[start:stop].keys():
            # positive interaction
            user_input.append(u)
            item_input.append(i)
            labels.append(1) # <- (Implicit ratings)
            # negative interactions
            for t in range(num_negatives): # <- num_negatives is a hyperparameter
                # randomly select an interaction; check if negative
                j = np.random.randint(num_items)
                while (u,j) in dok_train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        self.batch_counter += 1
        
        user_input = torch.tensor(user_input, dtype=torch.int32, device=self.device)
        item_input = torch.tensor(item_input, dtype=torch.int32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        labels = labels.reshape((labels.shape[0],1))
        yield user_input, item_input, labels
    
    def forward(self, user_input, item_input=None):
        raise NotImplementedError("Forward function not implemented.")

    def fit(self, epochs=30, batch_size=1024, learning_rate=0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) # <- The optimizer can be (additionally) considered as a hyperparameter
        for i in range(epochs):
            for user_input, item_input, labels in self._data_generator(batch_size):
                optimizer.zero_grad()
                predictions = self.forward(user_input, item_input)
                loss = torch.nn.BCELoss().to(self.device) # <- The loss function can be (additionally) considered as a hyperparameter
                loss = loss(predictions, labels.float())
                loss.backward()
                optimizer.step()
            self._print("Epoch {} finished. Loss: {}".format(i, loss.item()))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        step = user_id_array.shape[0]
        
        if items_to_compute is None:
            items_to_compute = np.arange(self.URM_train.shape[1], dtype=np.int32)
        
        predictions = np.empty((step,items_to_compute.shape[0]))
        for item in items_to_compute:
            with torch.no_grad():
                predictions[:, item] = self.forward(
                    torch.tensor(user_id_array),
                    torch.tensor(
                        np.ones(step, dtype=np.int32) * item)
                    ).cpu().detach().numpy().ravel()
        return predictions
    