import random
import pickle
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--output_file")
parser.add_argument("--dataset_file")
parser.add_argument("--matrix_file")

args = parser.parse_args()

data = open(args.dataset_file, "r").readlines()
sentences = [d.strip() for d in data]

count = 0
output = pickle.load(open(args.matrix_file +".pkl", "rb"))

count+=1
selected = []
print(sentences)
matrix = output['vanilla']
matrix[matrix<0] = 0 
relevance = []
surprise = output['surprise']
for idx in range(len(sentences)):
    relevance.append(sum(matrix[idx]))

penalty = [0 for i in range(len(sentences))]

for j in range(1, 4):
    selected = []
    summary = ""
    for k in range(j):
        maxIdx = -1
        maxVal = -float('inf')
        for i in range(len(sentences)):
            temp = np.dot([-1, 1], [penalty[i], relevance[i]])
            if temp > maxVal and i not in selected:
                maxIdx = i
                maxVal = temp 

        for i in range(len(sentences)):
            penalty[i]+=matrix[i][maxIdx]

        selected.append(maxIdx)
    summary = ""
    print(selected)
    for i in sorted(selected):
        print(sentences[i])
        summary+= sentences[i]+"\n" 


    with open(args.output_file+str(j), "a") as f:
        f.write(summary+'\n')