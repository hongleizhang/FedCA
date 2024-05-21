import copy

import torch

from model.engine import Engine


class FCF(torch.nn.Module):
    def __init__(self, config):
        super(FCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.user_embedding = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)

        self.logistic = torch.nn.Sigmoid()

        # self._init_weight_()


    def setItemEmbeddings(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)

    def _init_weight_(self):
        torch.nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data /= (self.config['latent_dim'] ** 0.5)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
        self.user_embedding.weight.data /= (self.config['latent_dim'] ** 0.5)

    def forward(self, item_indices):
        item_embeddings = self.item_embeddings(item_indices)
        user_embedding = self.user_embedding.weight
        rating = torch.mul(user_embedding, item_embeddings)
        rating = torch.sum(rating, dim=1)
        rating = self.logistic(rating)

        return rating

class FedNCF(torch.nn.Module):
    def __init__(self, config):
        super(FedNCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.user_embedding = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)

        self.mlp_weights = torch.nn.ModuleList()

        for idx, (in_size, out_size) in enumerate(zip(self.config['mlp_layers'][:-1], self.config['mlp_layers'][1:])):
            self.mlp_weights.append(torch.nn.Linear(in_size, out_size))

        self.logistic = torch.nn.Sigmoid()


    def setItemEmbeddings(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)

    def setMLPweights(self, mlp_weights):
        self.mlp_weights = copy.deepcopy(mlp_weights)


    def forward(self, item_indices):
        item_embeddings = self.item_embeddings(item_indices)
        user_embedding = self.user_embedding.weight
        repeat_num = item_embeddings.size(0)
        user_embedding = user_embedding.repeat(repeat_num, 1)
        vector = torch.cat([user_embedding, item_embeddings], dim=-1)
        for idx, _ in enumerate(range(len(self.mlp_weights))):
            vector = self.mlp_weights[idx](vector)
            if idx != len(self.mlp_weights)-1:
                vector = torch.nn.ReLU()(vector)
        rating = self.logistic(vector)

        return rating

class ModelEngine(Engine):
    """Engine for training & evaluating FCF model"""

    def __init__(self, config):
        model_name = config['backbone']
        self.model = eval(model_name)(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(ModelEngine, self).__init__(config)
        print(self.model)
