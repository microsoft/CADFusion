import json
import re
import argparse

"""
We did not implement the Hungarian matching algorithm from text2cad, but provided a vanilla matching for f1. It is because
 1. We argue that CAD scenarios are too complicated to be evaluated with a simple matching algorithm, especially when performed on the primitive level. Moreover, matching every primitive exactly is against the goal of our framework which attempt to encourage CAD models generate visually correct objects instead of accurate primitives compared to the ground truth.
 2. In our exploration, discrepancies on the number of primitives between model generation and the ground truth usually indicates the entire failure of the sketch so that using any of the algorithm does not affect the final evaluation result anyway.
 3. Our evaluation is a lower bound of the performance of the model on the matching algorithm, therefore it does not affect the overall integrety of our framework.

We encourage users to implement their own matching algorithm if they want to evaluate the model with a more strict metric.
"""

parser = argparse.ArgumentParser(description='Evaluate F1 scores for generated sketches.')
parser.add_argument('--test-path', type=str, default='data/sl_data/test.jsonl', help='Path to the JSONL file containing test data')    
parser.add_argument('--file_path', type=str, required=True, help='Path to the JSONL file containing generated sketches.')
args = parser.parse_args()
file_path = args.file_path
data_path = args.test_path
with open(data_path, 'r') as f:
    data = json.load(f)
    
def find_f1(ground_truth, pred, token):
    num_tok_gt = len(re.findall(token, ground_truth))
    num_tok_pred = len(re.findall(token, pred))
    # print(num_tok_gt, num_tok_pred)
    min_tok = min(num_tok_gt, num_tok_pred)
    if min_tok <= 0:
        return -1
    tok_recall = min_tok / num_tok_gt
    tok_precision = min_tok / num_tok_pred
    tok_f1 = 2 * tok_recall * tok_precision / (tok_recall + tok_precision)
    return tok_f1

with open(file_path, 'r') as f:
    gen = json.load(f)
line = []
arc = []
circle = []
ext = []
for i in range(1000):
    ground_truth = data[i]['output']
    pred = gen[i]['output']
    ext_f1 = find_f1(ground_truth, pred, r'<extrude_end>')
    if ext_f1 > 0:
        ext.append(ext_f1)

    skext_gt = ground_truth.split('<extrude_end>')[:-1]
    skext_pred = pred.split('<extrude_end>')[:-1]
    min_len_skext = min(len(skext_gt), len(skext_pred))
    if min_len_skext == 0:
        continue
    line_f1 = 0
    arc_f1 = 0
    circle_f1 = 0 
    for gt, pr in zip(skext_gt, skext_pred):
        line_f1 += find_f1(gt, pr, r'line.*?<curve_end>')
        arc_f1 += find_f1(gt, pr, r'arc.*?<curve_end>')
        circle_f1 += find_f1(gt, pr, r'circle.*?<curve_end>')
        
    line_f1 = line_f1 / min_len_skext
    arc_f1 = arc_f1 / min_len_skext
    circle_f1 = circle_f1 / min_len_skext
    if line_f1 > 0:
        line.append(line_f1)
    if arc_f1 > 0:
        arc.append(arc_f1)
    if circle_f1 > 0:
        circle.append(circle_f1)
line_avg = sum(line) / len(line)
arc_avg = sum(arc) / len(arc)
circle_avg = sum(circle) / len(circle)
avgf1 = (line_avg + arc_avg + circle_avg) / 3
print(file_path, line_avg, arc_avg, circle_avg, avgf1, sum(ext) / len(ext))