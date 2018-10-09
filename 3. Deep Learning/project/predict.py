import argparse
from model_utils import load_from_checkpoint, predict, get_category_labels

def_top_k = 5
def_category_names = './cat_to_name.json'

parser=argparse.ArgumentParser()

parser.add_argument('image_filepath', metavar='image_filepath', type=str, help='Path to image to predict on.')
parser.add_argument('checkpoint_filepath', metavar='checkpoint_filepath', type=str, help='Path to model checkoint to predict with.')
parser.add_argument('--top_k', nargs='?', default=def_top_k, type=int, help='Number of best predictions to show.')
parser.add_argument('--category_names', nargs='?', default=def_category_names, type=str, help=f'Filepath to mappging of labels to be used in place of numerical categories. Default is {def_category_names}.')
parser.add_argument('--gpu', action='store_true', help='Pass this flag to use GPU if available.')

args=parser.parse_args()
print(args)

model = load_from_checkpoint(args.checkpoint_filepath)

probs, classes = predict(args.image_filepath, model, args.gpu, args.top_k)

labal_map = get_category_labels(args.category_names)
labels = labels = [labal_map[cls + 1] for cls in classes]
print(probs)
print(classes)
print(labels)
