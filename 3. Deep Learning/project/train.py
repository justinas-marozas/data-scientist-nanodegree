import argparse
import datetime
from model_utils import build_data_loaders, build_model, train, save_checkpoint

def_save_dir = './'
def_arch = 'densenet-121'
def_lr = 0.001
def_hidden_units = [2048, 1024]
def_output_units = 102
def_epochs = 3
def_gpu = False

parser=argparse.ArgumentParser()

parser.add_argument('data_dir', metavar='data_dir', type=str, help='Data directory for model training.')
parser.add_argument('--save_dir', nargs='?', default=def_save_dir, type=str, help='Directory to save the model checkpoints. Current directory as default.')
parser.add_argument('--arch', nargs='?', default=def_arch, type=str, help=f'Model architecture. {def_arch} as default.')
parser.add_argument('--learning_rate', nargs='?', default=def_lr, type=float, help=f'Model learning rate. Between 0 and 1. Default {def_lr}.')
parser.add_argument('--hidden_units', nargs='+', default=def_hidden_units, type=int, help=f'Sizes of hidden layers in model classifier. Can pass multiple arguments. Default: {" ".join([str(_) for _ in def_hidden_units])}.')
parser.add_argument('--output_units', nargs='?', default=def_output_units, type=int, help=f'Size of output layer, or number of prediction classes. Default is {def_output_units}.')
parser.add_argument('--epochs', nargs='?', default=def_epochs, type=int, help=f'Number of training epochs to run. Default is {def_epochs}.')
parser.add_argument('--gpu', action='store_true', help='Pass this flag to use GPU if available.')

args=parser.parse_args()
print(args)

loaders = build_data_loaders(args.data_dir)
model = build_model(args.arch, args.hidden_units, args.output_units)
best_model = train(model, args.epochs, args.learning_rate, args.gpu, loaders)
now = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%dT%H%M%S')
save_checkpoint(f'{args.save_dir}/checkpoint-{args.arch}-{now}.pth', best_model, args.arch)
