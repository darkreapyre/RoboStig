from __future__ import print_function
import argparse
parser = argparse.ArgumentParser(description='Train on CIFAR-10.')
parser.add_argument('--training', type=str, help='Path to CIFAR-10 training data')
parser.add_argument('--learning-rate', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=1e-6,
                    help='Learning rate decay')
parser.add_argument('--epochs', type=int, default=20,
                   help='Number of epochs to train')
parser.add_argument('--augment-data', type=bool, default=True,
                    help='Whether to augment data [TRUE | FALSE]')
parser.add_argument('--output_data_dir', type=str, help='Path to model output')
args, _ = parser.parse_known_args()
#args = parser.parse_args()
print("\n")
print(args)