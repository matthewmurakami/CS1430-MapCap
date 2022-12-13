import random
import pickle
import numpy as np

def main(outfile, data_file, matrix_file):
    data = open(data_file, "r").readlines()
    sentences = [d.strip() for d in data]

    count = 0
    output = pickle.load(open(matrix_file, "rb"))

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
        for i in sorted(selected):
            summary+= sentences[i]+"\n" 


        with open(outfile+str(j), "a") as f:
            f.write(summary+'\n')