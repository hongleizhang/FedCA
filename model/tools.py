import copy

import cvxpy as cp
import numpy as np
import torch
from sklearn.decomposition import PCA


def aggregateByComposite(client_params, graph_matrix, config):
    # aggregated client params dict
    aggregated_client_params = copy.deepcopy(client_params)
    for user in aggregated_client_params.keys():
        for key in aggregated_client_params[user].keys():
            aggregated_client_params[user][key] = torch.zeros_like(aggregated_client_params[user][key])

    for user in client_params.keys():
        aggregation_weight_vector = graph_matrix[user]
        # aggregate client params by graph
        for neighbor_user in client_params.keys():
            neighbor_param = client_params[neighbor_user]['item_embeddings.weight'].data
            neighbor_weight = aggregation_weight_vector[neighbor_user]
            aggregated_client_params[user]['item_embeddings.weight'].data += neighbor_weight * neighbor_param
            if config['backbone'] == 'FedNCF':
                for layer in range(len(config['mlp_layers']) - 1):
                    aggregated_client_params[user]['mlp_weights.' + str(layer) + '.weight'].data += \
                    client_params[neighbor_user]['mlp_weights.' + str(layer) + '.weight'].data * neighbor_weight
                    aggregated_client_params[user]['mlp_weights.' + str(layer) + '.bias'].data += \
                    client_params[neighbor_user]['mlp_weights.' + str(layer) + '.bias'].data * neighbor_weight

    return aggregated_client_params


def update_composite_matrix_neighbor(graph_matrix, client_params, init_param, principal_list, client_sample_num,
                                     lambda_1,
                                     lambda_2):
    index_clientid = list(client_params.keys())
    if lambda_2 == 0:
        model_similarity_matrix = torch.zeros_like(graph_matrix)
    else:
        model_similarity_matrix = calculate_l2_similarity(client_params, init_param)
    if lambda_1 == 0:
        model_complementary_matrix = torch.zeros_like(graph_matrix)
    else:
        model_complementary_matrix = calculate_complementary(client_params, principal_list)

    graph_matrix = optimizing_composite_matrix_neighbor(graph_matrix, index_clientid, model_similarity_matrix,
                                                        model_complementary_matrix, client_sample_num, lambda_1,
                                                        lambda_2)
    return graph_matrix, -model_similarity_matrix, model_complementary_matrix


def optimizing_composite_matrix_neighbor(graph_matrix, client_id, client_similarity_matrix, client_complementary_matrix,
                                         client_sample_num, lambda_1, lambda_2):
    fed_avg_freqs = calculate_client_weights(client_sample_num)

    n = client_similarity_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(n):
        model_complementary_vector = client_complementary_matrix[i]
        model_difference_vector = client_similarity_matrix[i]
        s = model_difference_vector.numpy()
        c = model_complementary_vector.numpy()

        q = - lambda_1 * 2 * c + lambda_2 * 2 * s - 2 * p

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()
        graph_matrix[client_id[i], client_id] = torch.Tensor(x.value)

    return graph_matrix


def calculate_l2_similarity(client_params, init_param):
    # define similarity matrix
    n = len(client_params)
    client_similarity_matrix = torch.zeros(n, n)
    client_id = list(client_params.keys())

    pca = PCA(n_components=1, svd_solver='randomized', random_state=42)
    init_params = pca.fit_transform(init_param['item_embeddings.weight'].data.cpu().numpy())
    for i in range(len(client_id)):
        user_param_i = client_params[client_id[i]]['item_embeddings.weight'].data.cpu().numpy()
        # get user params
        user_param_i = pca.fit_transform(user_param_i) - init_params
        user_param_i = torch.tensor(user_param_i).view(user_param_i.shape[1], -1)
        for j in range(i, len(client_id)):
            user_param_j = client_params[client_id[j]]['item_embeddings.weight'].data.cpu().numpy()
            user_param_j = pca.fit_transform(user_param_j) - init_params
            user_param_j = torch.tensor(user_param_j).view(user_param_j.shape[1], -1)

            distance = torch.norm(user_param_i - user_param_j)  # Efficient training for gpu
            similarities = - 1 / (1 + distance)

            # calculate average similarity
            average_similarity = similarities.mean()

            if average_similarity < - 0.9:
                average_similarity = -1.0

            client_similarity_matrix[i][j] = average_similarity
            client_similarity_matrix[j][i] = average_similarity

    return client_similarity_matrix


def calculate_complementary(client_params, principal_list):
    # define complementary matrix
    n = len(client_params)
    client_complementary_matrix = torch.zeros(n, n)
    client_id = list(client_params.keys())
    # dimension k
    k = principal_list[0].shape[0]

    for i in range(len(client_id)):
        for j in range(i, len(client_id)):
            phi = compute_principal_angles(principal_list[i], principal_list[j])
            principal_angle = np.cos((1 / k) * np.sum(phi))
            client_complementary_matrix[i][j] = principal_angle
            client_complementary_matrix[j][i] = principal_angle

    return client_complementary_matrix


def calculate_client_weights(client_sample_num):
    if not client_sample_num:
        raise ValueError("The 'client_sample_num' dictionary cannot be empty.")

    total_samples = sum(client_sample_num.values())
    if total_samples == 0:
        raise ValueError("Total sample number must be greater than zero to calculate proportions.")
    client_proportions = {client: samples / total_samples for client, samples in client_sample_num.items()}

    return client_proportions


def get_principal(client_train_data, client_param_dict, k_principal):
    ### ??? sample_num * dimension --> k_principal * dimension
    extracted_item_id = torch.LongTensor(client_train_data[1])
    client_item_embedding = client_param_dict['item_embeddings.weight']
    extracted_item_feature = client_item_embedding[extracted_item_id]
    extracted_item_feature = extracted_item_feature.detach().cpu().numpy()

    pca = PCA(n_components=k_principal)
    pca.fit_transform(extracted_item_feature)
    orthogonal_basis = pca.components_

    return orthogonal_basis


def compute_principal_angles(A, B):
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"

    k = A.shape[0]
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_B = np.linalg.norm(B, axis=1)
    dot_product = np.dot(A, B.T)
    cosine_matrix = dot_product / (norm_A * norm_B)
    cos_phi_values = []

    for _ in range(k):
        i, j = np.unravel_index(np.argmax(cosine_matrix, axis=None), cosine_matrix.shape)
        cos_phi_values.append(cosine_matrix[i, j])
        cosine_matrix[i, :] = -np.inf
        cosine_matrix[:, j] = -np.inf
    phi = np.arccos(np.clip(cos_phi_values, -1, 1))

    return phi


def weight_client_server(user, client_model_params, server_model_param, map, model_dict, config):
    client_param_dict = copy.deepcopy(model_dict)

    client_items_data = client_model_params[user]['item_embeddings.weight'].data

    if user in map:
        server_items_data = server_model_param[map[user]]['item_embeddings.weight'].data
        client_param_dict['item_embeddings.weight'].data = config['interpolation'] * client_items_data + (
                    1 - config['interpolation']) * server_items_data
    else:
        client_param_dict['item_embeddings.weight'].data = client_items_data

    # for mlp layers param
    if config['backbone'] == 'FedNCF':
        for layer in range(len(config['mlp_layers']) - 1):
            client_mlp_weight = client_model_params[user]['mlp_weights.' + str(layer) + '.weight'].data
            client_mlp_bias = client_model_params[user]['mlp_weights.' + str(layer) + '.bias'].data

            if user in map:
                server_mlp_weight = server_model_param[map[user]]['mlp_weights.' + str(layer) + '.weight'].data
                server_mlp_bias = server_model_param[map[user]]['mlp_weights.' + str(layer) + '.bias'].data
                client_param_dict['mlp_weights.' + str(layer) + '.weight'].data = config[
                                                                                      'interpolation'] * client_mlp_weight + (
                                                                                              1 - config[
                                                                                          'interpolation']) * server_mlp_weight
                client_param_dict['mlp_weights.' + str(layer) + '.bias'].data = config[
                                                                                    'interpolation'] * client_mlp_bias + (
                                                                                            1 - config[
                                                                                        'interpolation']) * server_mlp_bias

            else:
                client_param_dict['mlp_weights.' + str(layer) + '.weight'].data = client_mlp_weight
                client_param_dict['mlp_weights.' + str(layer) + '.bias'].data = client_mlp_bias

    return client_param_dict


def sub_matrix_shift(matrix):
    matrix[matrix.abs() < 1e-20] = 0
    row_sums = matrix.sum(dim=1)
    matrix = matrix / row_sums.view(-1, 1)

    return matrix
