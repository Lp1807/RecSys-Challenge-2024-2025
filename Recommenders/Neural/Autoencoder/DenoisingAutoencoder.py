import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Recommenders.Neural.DeepLearningRecommender import DeepLearningRecommender

class DenoisingAutoencoder(DeepLearningRecommender):

    RECOMMENDER_NAME = """DENOISING_AUTOENCODER"""
    def __init__(self, URM_train, encoding_dim=69, noise_p=0.01, verbose=True):
        super().__init__(URM_train, verbose)
        self.noise_p = noise_p
        num_items = URM_train.shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 420),
            nn.ReLU(),
            nn.Linear(420, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 420),
            nn.ReLU(),
            nn.Linear(420, num_items)
        )
        self.to(self.device)
    
    # override: both input and label are batches of user profiles
    def _data_generator(self, batch_size):
        row_idx = np.arange(self.URM_train.shape[0])
        for start in range(0, len(row_idx), batch_size):
            end = min(len(row_idx), start + batch_size)
            user_input = torch.tensor(self.URM_train[row_idx[start:end],:].toarray(), dtype=torch.float32, device=self.device)
            labels = user_input
            _ = "placeholder"
            yield user_input, _, labels

    def forward(self, user_input, item_input=None):
        # assert(item_input == None, "Item input not needed")
        noisy_input = self._add_noise(user_input)
        encoded = self.encoder(noisy_input)
        reconstructed = self.decoder(encoded)
        return reconstructed

    # override: evaluator passes user profile ids as inputs, we need the
    #           full profiles for the forward function to work properly
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_profiles = self.URM_train[user_id_array, :]

        if items_to_compute is not None:
            mask = np.zeros(self.URM.shape[1], dtype=np.int32)
            mask[items_to_compute] = 1
            user_profiles = user_profiles[:, mask]

        with torch.no_grad():
            predictions = self.forward(torch.tensor(user_profiles.toarray(), dtype=torch.float32, device=self.device))

        return predictions.cpu().detach().numpy()

    def _add_noise(self, x):
        zeros_mask = np.random.choice([False,True], size=x.shape, p=[1-self.noise_p, self.noise_p])
        ones_mask = np.random.choice([False,True], size=x.shape, p=[1-self.noise_p, self.noise_p])
        x[zeros_mask] = 0
        x[ones_mask] = 1
        return x