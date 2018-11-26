import json
import argparse
from help_functions import load_checkpoint, process_image, predict

parser = argparse.ArgumentParser(description='Predict inputs')
parser.add_argument('image_path', action='store', type=str)
parser.add_argument('checkpoint', action='store', type=str)
parser.add_argument('--top_k', action='store', default=1, type=int)
parser.add_argument('--category_names', action='store', default='NA', type=str)
parser.add_argument('--gpu', action='store_true')
inputs = parser.parse_args()
#Load model
model = load_checkpoint(inputs.checkpoint)
# Set it to eval and to cpu or gpu based on input
if inputs.gpu:
    device = 'cuda'
else:
    device = 'cpu'
model.to(device)
model.eval()
ps, classes = predict(inputs.image_path, model, device, inputs.top_k)

if inputs.category_names != 'NA':
    with open(inputs.category_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[x] for x in classes]
else:
    flower_names = classes
for flower,probability in zip(flower_names, ps):
    print('%.2f probability it is a %s' % (probability, flower))