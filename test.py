"""Evaluate a trained FedCA or FedAvg checkpoint on its test split."""

import argparse
import logging
import os

import torch

from model.model import ModelEngine
from utils.data import SampleGenerator
from utils.utils import loadData, setSeed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', type=str, default='fedca', choices=['fedca', 'fedavg'],
                        help='method name used to locate the default checkpoint')
    parser.add_argument('--model_path', type=str, default=None,
                        help='checkpoint path (default: checkpoints/<method>.pt)')
    parser.add_argument('--datasets_dir', type=str, default='./datasets',
                        help='directory containing the dataset folders')
    parser.add_argument('--use_cuda', action='store_true', help='run evaluation on a CUDA device')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--top_k', type=int, default=None,
                        help='override the checkpoint evaluation cutoff')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_path is None:
        args.model_path = os.path.join('checkpoints', '{}.pt'.format(args.method))
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(args.model_path))
    if args.use_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA evaluation was requested, but CUDA is not available')
    if args.use_cuda:
        torch.cuda.set_device(args.device_id)

    map_location = 'cuda:{}'.format(args.device_id) if args.use_cuda else 'cpu'
    checkpoint = torch.load(args.model_path, map_location=map_location)
    config = checkpoint['config'].copy()
    config.setdefault('method', 'fedca')
    config['use_cuda'] = args.use_cuda
    config['device_id'] = args.device_id
    if args.top_k is not None:
        config['top_k'] = args.top_k

    setSeed(config['seed'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

    ratings, num_users, num_items = loadData(
        args.datasets_dir, config['dataset'], config, config['data_file'])
    if (num_users, num_items) != (config['num_users'], config['num_items']):
        raise ValueError(
            'Dataset shape ({}, {}) does not match checkpoint shape ({}, {})'.format(
                num_users, num_items, config['num_users'], config['num_items']))

    sample_generator = SampleGenerator(ratings=ratings, config=config)
    engine = ModelEngine(config)
    engine.loadCheckpoint(checkpoint)
    hr, ndcg = engine.federatedEvaluate(sample_generator.test_data)
    logging.info('Test HR@%d = %.4f, NDCG@%d = %.4f',
                 config['top_k'], hr, config['top_k'], ndcg)


if __name__ == '__main__':
    main()
