"""
    Some handy functions for pytroch model training ...
"""
import logging
import os

import numpy as np
import torch


# Checkpoints
def saveCheckPoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resumeCheckPoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(
                                device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def initLogging(log_file_name):
    """Init for logging"""
    import logging
    import coloredlogs

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=log_file_name,
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    coloredlogs.install()


def setSeed(seed=0):
    """Set all random seeds"""

    import random
    import numpy as np
    import torch

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def datasetFilter(ratings, min_items=5):
    """
            Only keep the data useful, which means:
                - all ratings are non-zeros
                - each user rated at least {self.min_items} items
            :param ratings: pd.DataFrame
            :param min_items: the least number of items user rated
            :return: filter_ratings: pd.DataFrame
            """

    # filter unuseful data
    ratings = ratings[ratings['rating'] > 0]

    # only keep users who rated at least {self.min_items} items
    user_count = ratings.groupby('uid').size()
    user_subset = np.in1d(ratings.uid, user_count[user_count >= min_items].index)
    filter_ratings = ratings[user_subset].reset_index(drop=True)

    del ratings

    return filter_ratings


def loadData(path, dataset, config, file_name='ratings.dat'):
    import os
    import pandas as pd

    assert dataset in ['ml-100k', 'ml-1m', 'filmtrust', 'microlens'], "请使用指定数据集：ml-100k, ml-1m, filmtrust, microlens"

    dataset_file = os.path.join(path, dataset, file_name)

    min_rates = 10

    if dataset == "ml-100k":
        ratings = pd.read_csv(dataset_file, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')

    elif dataset == "ml-1m":
        ratings = pd.read_csv(dataset_file, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')

    elif dataset == "filmtrust":
        ratings = pd.read_csv(dataset_file, sep=" ", header=None, names=['uid', 'mid', 'rating'],
                              engine='python')

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

    elif dataset == "microlens":
        ratings = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')

    else:
        ratings = pd.DataFrame()


    ratings = datasetFilter(ratings, min_rates)

    # Reindex user id and item id
    user_id = ratings[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

    item_id = ratings[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

    # 先根据user进行升序排序，然后再每个用户上按照时间戳进行升序排序
    ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].groupby('userId', group_keys=False).apply(lambda x: x.sort_values('timestamp'))

    num_users, num_items = print_statistics(ratings)

    return ratings, num_users, num_items



def print_statistics(ratings):
    """print the statistics of the dataset, and return the number of users and items"""
    maxs = ratings.max()
    num_interactions = len(ratings)
    sparsity = 1 - num_interactions / ((maxs['userId'] + 1) * (maxs['itemId'] + 1))

    user_average_items = ratings.groupby('userId')['itemId'].count().sum()/(maxs['userId'] + 1)
    item_avearge_users = ratings.groupby('itemId')['userId'].count().sum()/(maxs['itemId'] + 1)

    logging.info('The number of users: {}, and of items: {}.'.format(int(maxs['userId'] + 1), int(maxs['itemId'] + 1)))
    logging.info('There are total {} interactions, the sparsity is {:.2f}%.'.format(num_interactions, sparsity * 100))
    logging.info('The averaged number of items interacted by user : {:.2f}, and by item: {:.2f}.'.format(user_average_items, item_avearge_users))

    return int(maxs['userId'] + 1), int(maxs['itemId'] + 1)

def format_arg_str(arg_dict, exclude_lst: list, max_len=20) -> str:
    linesep = os.linesep
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len - 3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str