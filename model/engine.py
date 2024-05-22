import copy
import random

import torch
from torch.utils.data import DataLoader

from utils.data import UserItemRatingDataset
from utils.metrics import MetronAtK
from .tools import aggregateByComposite
from .tools import get_principal, sub_matrix_shift, weight_client_server, update_composite_matrix_neighbor


class Engine(object):
    """
    Meta Engine for training & evaluating our model

    Note: Subclass should implement self.model!

    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.server_model_param = {}
        self.client_model_params = {}
        self._metron = MetronAtK(top_k=self.config['top_k'])

    def instanceUserTrainLoader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True), len(dataset)

    def federatedTrainSingleBatch(self, model_client, batch_data, optimizer):
        """train a batch and return an updated model."""
        from model.loss import Loss

        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        model_loss = Loss(self.config)

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            model_loss = model_loss.cuda()

        # update score function.
        optimizer.zero_grad()
        ratings_pred = model_client(items)
        loss = model_loss(ratings_pred.view(-1), ratings)
        loss.backward()

        optimizer.step()

        if self.config['use_cuda']:
            loss = loss.cpu()

        return model_client, loss.item()

    def aggregateParamsFromClients(self, client_params, **kwargs):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # KeyError
        if 'graph_matrix' not in kwargs or kwargs['graph_matrix'] is None:
            raise ValueError("The 'graph_matrix' argument cannot be empty.")
        # inite structure
        self.server_model_param = copy.deepcopy(client_params)
        # aggregating
        self.server_model_param = aggregateByComposite(client_params, kwargs["graph_matrix"], self.config)
        pass

    def federatedTrainOneRound(self, train_data, item_embeddings, mlp_weights, iteration):
        """sample users participating in single round."""

        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)

        # store users' model parameters of current round.
        participant_params = {}
        # store all the users' train loss/
        losses = {}
        # store all the clients' sample num
        client_sample_num = {}
        # principal list for calculating complementary matrix
        principal_list = []

        # perform model update for each participated user.
        for user in participants:
            loss = 0

            # for the first round, client models copy initialized parameters directly.
            client_model = copy.deepcopy(self.model)
            client_model.setItemEmbeddings(item_embeddings)
            if self.config['backbone'] == 'FedNCF':
                client_model.setMLPweights(mlp_weights)

            # for other rounds, client models receive updated item embedding and score function from server.
            if iteration != 0:
                client_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        client_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)

                client_param_dict = weight_client_server(user, self.client_model_params, self.server_model_param,
                                                         self.agg_participant_index_map, client_param_dict, self.config)

                if self.config['use_cuda']:
                    for key in client_param_dict.keys():
                        client_param_dict[key] = client_param_dict[key].cuda()

                client_model.load_state_dict(client_param_dict)

            # Defining optimizers
            if self.config['backbone'] == 'FCF':
                optimizer = torch.optim.Adam([
                    {'params': client_model.user_embedding.parameters(), 'lr': self.config['lr_embedding']},
                    {'params': client_model.item_embeddings.parameters(), 'lr': self.config['lr_embedding']}
                ])
            elif self.config['backbone'] == 'FedNCF':
                optimizer = torch.optim.Adam([
                    {'params': client_model.user_embedding.parameters(), 'lr': self.config['lr_embedding']},
                    {'params': client_model.item_embeddings.parameters(), 'lr': self.config['lr_embedding']},
                    {'params': client_model.mlp_weights.parameters(), 'lr': self.config['lr_structure'],
                     'weight_decay': self.config['weight_decay']}
                ])

            # load current user's training data and instance a train loader.
            user_train_data = [train_data[0][user], train_data[1][user], train_data[2][user]]
            # record the current user's sample num
            user_dataloader, user_sample_num = self.instanceUserTrainLoader(user_train_data)
            client_sample_num[user] = user_sample_num

            client_model.train()
            sample_num = 0
            # update client model.
            client_losses = []
            for epoch in range(self.config['local_epoch']):

                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    client_model, client_loss = self.federatedTrainSingleBatch(client_model, batch, optimizer)
                    loss += client_loss * len(batch[0])
                    sample_num += len(batch[0])

                losses[user] = loss / sample_num
                client_losses.append(loss / sample_num)

                # check convergence, avoid division by zero
                if epoch > 0 and abs(client_losses[epoch] - client_losses[epoch - 1]) / abs(
                        client_losses[epoch - 1] + 1e-5) < self.config['threshold']:
                    break

            # obtain client model parameters,
            self.client_model_params[user] = copy.deepcopy(client_model.state_dict())

            user_principal = get_principal(user_train_data, client_model.state_dict(), self.config['k_principal'])
            principal_list.append(user_principal)

            # store client models' local parameters for global update.
            participant_params[user] = copy.deepcopy(self.client_model_params[user])

            # delete all user-related data
            del participant_params[user]['user_embedding.weight']

        # aggregate client models in server side.
        agg_num = int(len(participants) * self.config["agg_clients_ratio"])
        agg_participants = random.sample(list(participant_params.keys()), agg_num)
        # map for user and idx
        participant_index_map = {participant: index for index, participant in enumerate(participant_params.keys())}
        graph_indices = [participant_index_map[participant] for participant in agg_participants]
        self.graph_matrix = torch.ones(len(agg_participants), len(agg_participants)) / (len(agg_participants))
        agg_principal_list = [principal_list[idx] for idx in graph_indices]
        agg_participants_params = {idx: participant_params[user] for idx, user in enumerate(agg_participants)}
        agg_client_sample_num = {idx: client_sample_num[user] for idx, user in enumerate(agg_participants)}
        self.agg_participant_index_map = {user: idx for idx, user in enumerate(agg_participants)}

        self.graph_matrix, self.model_smi, self.model_comp = update_composite_matrix_neighbor(self.graph_matrix,
                                                                                              agg_participants_params,
                                                                                              self.model.state_dict(),
                                                                                              agg_principal_list,
                                                                                              agg_client_sample_num,
                                                                                              self.config['alpha'],
                                                                                              self.config['beta'])
        self.graph_matrix = sub_matrix_shift(self.graph_matrix)
        self.aggregateParamsFromClients(agg_participants_params, client_sample_num=agg_client_sample_num,
                                        graph_matrix=self.graph_matrix)

        return losses

    @torch.no_grad()
    def federatedEvaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]

        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()

        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None

        for user in range(self.config['num_users']):
            user_model = copy.deepcopy(self.model)

            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data
            else:
                user_param_dict = copy.deepcopy(self.model.state_dict())

            if user in self.agg_participant_index_map:
                user_param_dict = weight_client_server(user, self.client_model_params, self.server_model_param,
                                                       self.agg_participant_index_map, user_param_dict, self.config)

            user_model.load_state_dict(user_param_dict)

            user_model.eval()

            # obtain user's positive test information.
            test_item = test_items[user: user + 1]
            # obtain user's negative test information.
            negative_item = negative_items[user * 99: (user + 1) * 99]
            # perform model prediction.
            test_score = user_model(test_item)
            negative_score = user_model(negative_item)

            if user == 0:
                test_scores = test_score
                negative_scores = negative_score
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))

        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        hr, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()

        return hr, ndcg
