import os
import sys

from models.gpt2 import GPT2
import math
import numpy as np
import pickle
import spacy
import argparse


def get_probabilities(articles, model):
    """
    Given a batch of articles (can be any strings) run a forward pass on GPT2 and obtain word probabilities for the same
    """
    article_splits = [article.split(" ") for article in articles]
    payload = model.get_probabilities(articles, topk = 20)
    res = [[] for i in range(len(articles))]
    for t, article in enumerate(articles):
        context = ""
        idx = 0
        chain = False
        next_word = ""
        article_words = article_splits[t]
        #print(article, article_words)
        word_probability = 1.0
        gt_count = 0
        idx+=1
        for i, word in enumerate(payload["context_strings"][t][:-1]):
            context = context+" "+word
            probability = payload['real_probs'][t][i]#[1]
            next_word_fragment = payload["context_strings"][t][i+1]

            next_word += next_word_fragment
            #print(next_word, article_words[gt_count])
            if next_word == article_words[gt_count]:
                chain = False
                gt_count+=1
            else:
                chain = True

            word_probability *= probability
            assert word_probability <= 1.0, print(word_probability, context)
            if chain == False:      
                #print("Word Probability: ", word_probability, next_word)
                res[t].append(word_probability)
                word_probability = 1.0 
                next_word = ""
            #print(gt_count, len(article_words))
            if gt_count == len(article_words):
                break
    return res


def get_npmi_matrix(sentences, model, method = 1, batch_size = 1):
    """
    Accepts a list of sentences of length n and returns 3 objects:
    - Normalised PMI nxn matrix - temp
    - PMI nxn matrix - temp2
    - List of length n indicating sentence-wise surprisal i.e. p(sentence) - p 

    To optimize performance, we do the forward pass batchwise by assembling the batch and maintaining batch indices
    For each batch we call get_probabilities
    """
    temp = np.zeros((len(sentences), len(sentences)))
    temp2 = np.zeros((len(sentences), len(sentences)))
    batch_indices = {}
    batch = []
    batchCount = 0
    batchSize = batch_size

    c = 0
    p = []
    for i in range(len(sentences)):
        print(sentences[i])
        result = get_probabilities([sentences[i].strip()], model)
        try:
            p.append(sum([math.log(i) for i in result[0]]))
        except:
            print("Math domain error surprise", i)
            return temp, temp2, p
    for i in range(len(sentences)):
        print(i)
        for j in range(len(sentences)):
            if i==j: 
                temp[i][j] = -1
                temp2[i][j] = -1
                continue
            article = sentences[i].strip() + " "+ sentences[j].strip()
            print(article)
            batch_indices[str(i)+"-"+str(j)+"-"+str(len(sentences[i].split()))] = batchCount 
            batch.append(article)
            batchCount+=1
            
            if batchCount == batchSize or (i == len(sentences)-1 and j == len(sentences)-1):
                c+=1
                result = get_probabilities(batch, model)
                for key in batch_indices.keys():

                    idx_i, idx_j, idx_l = [int(idx) for idx in key.split("-")]
                    try:
                        pxy = sum([math.log(q) for q in result[batch_indices[key]][idx_l:]])
                        py = p[idx_j]
                        px = p[idx_i]
                    
                        temp[idx_i][idx_j] = (pxy - py)/(-1*(pxy+px))
                        temp2[idx_i][idx_j] = (pxy - py)
                    except ZeroDivisionError:
                        print("Zero division error ", idx_i, idx_j)
                        temp[idx_i][idx_j] = -1
                        temp2[idx_i][idx_j] = -1
                    except:
                        print("Math Domain Error", i, j)
                    if temp[idx_i][idx_j] > 1 or temp[idx_i][idx_j] < -1:
                        print("Normalise assert ", temp[idx_i][idx_j], idx_i, idx_j)
                batchCount = 0
                batch = []
                batch_indices = {}
    return temp, temp2, p


"""
Main iteration loop, creates matrices for each document in the dataset
"""

def main(input_file_path, out_path, device):

    data = open(input_file_path, "r").readlines()

    model = GPT2(device=device, location="")

    if os.path.exists(out_path):
        output = pickle.load(open(out_path,"rb"))
    else:
        output = {}

    normalised, vanilla, surprise = get_npmi_matrix(data, model) 

    output = {}
    output["vanilla"] = vanilla
    output["normalised"] = normalised
    output["surprise"] = surprise

    pickle.dump(output, open(out_path, "wb"))
