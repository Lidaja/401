import numpy as np
import sys
import argparse
import os
import json
import re
import string
import csv

fp = [word.rstrip().lower() for word in open('/u/cs401/Wordlists/First-person','r').readlines()]
sp = [word.rstrip().lower() for word in open('/u/cs401/Wordlists/Second-person','r').readlines()]
tp = [word.rstrip().lower() for word in open('/u/cs401/Wordlists/Third-person','r').readlines()]
cc = [word.rstrip().lower() for word in open('/u/cs401/Wordlists/Conjunct','r').readlines()]

rid = [word.rstrip() for word in open('/u/cs401/A1/feats/Right_IDs.txt','r').readlines()]
cid = [word.rstrip() for word in open('/u/cs401/A1/feats/Center_IDs.txt','r').readlines()]
lid = [word.rstrip() for word in open('/u/cs401/A1/feats/Left_IDs.txt','r').readlines()]
aid = [word.rstrip() for word in open('/u/cs401/A1/feats/Alt_IDs.txt','r').readlines()]

rfeat = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
cfeat = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
lfeat = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
afeat = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')

with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r') as f:
    bglReader = csv.reader(f)
    bglData = list(bglReader)
bglDict = {}
for i in range(1,len(bglData)):
    bglDict[bglData[i][1]] = [bglData[i][3],bglData[i][4],bglData[i][5]]

with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r') as f:
    wReader = csv.reader(f)
    wData = list(wReader)
wDict = {}
for i in range(1,len(wData)):
    wDict[wData[i][1]] = [wData[i][2],wData[i][5],wData[i][8]]

slang = list(set([word.rstrip().lower() for word in open('/u/cs401/Wordlists/Slang','r').readlines()] + [word.rstrip().lower() for word in open('/u/cs401/Wordlists/Slang2','r').readlines()]))

def counter(comment, featureType):
    temp = 0
    lowerComment = [c.lower() for c in comment]
    for word in featureType:
        temp+= lowerComment.count(word)
    return temp

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros((174,))

    splitComment = comment.split(" ")
    tags = [word.rsplit("/")[-1] for word in splitComment]
    words = [word.rsplit("/")[0] for word in splitComment]
    sentenceSplit = comment.split("\n")

    #First person
    feats[0] = counter(words,fp)

    #Second person
    feats[1] = counter(words,sp)

    #Third person
    feats[2] = counter(words,tp)

    #Conjunctions
    feats[3] = counter(words,cc)

    #Past verbs
    feats[4] = tags.count('VBN') + tags.count('VBD')

    #Future verbs
    feats[5] = tags.count('BES') + tags.count('MD')

    #Commas
    feats[6] = tags.count(',')
    in_waiting_room == 0 && total_customers <= customers_served+customers_angry
    #Multicharacter punctuation
    feats[7] = len(re.findall(re.compile("["+string.punctuation+"]{2,}")," ".join(words)))

    #Common nouns
    feats[8] = tags.count('NN') + tags.count('NNS')

    #Proper nouns
    feats[9] = tags.count('NNP') + tags.count('NNPS')

    #Adverbs
    feats[10] = tags.count('RB') + tags.count('RBR') + tags.count('RBS') + tags.count('RP')

    #Wh words
    feats[11] = tags.count('WDT') + tags.count('WP') + tags.count('WP$') + tags.count('WRB')

    feats[12] = counter(words,slang)

    for word in words:
        if len(word) >= 3 and word.isalpha() and word == word.upper():
            feats[13] += 1

    for sentence in sentenceSplit:
        wordSplit = sentence.split(" ")
        feats[14] += len(wordSplit)
    feats[14] = feats[14]*1.0/len(sentenceSplit)
    
    numTokens = 0
    lenTokens = 0
    for word in words:
        if word.isalnum():
            numTokens += 1
            lenTokens += len(word)
    if lenTokens > 0:
        feats[15] = lenTokens*1.0/numTokens

    feats[16] = len(sentenceSplit)

    AoA,IMG,FAM = [],[],[]
    for word in words:
        if word in bglDict:
            try:
                AoA.append(float(bglDict[word][0]))
            except:
                pass
            try:
                IMG.append(float(bglDict[word][1]))
            except:
                pass
            try:
                FAM.append(float(bglDict[word][2]))
            except:
                pass
    if len(AoA) > 0:
        feats[17] = np.mean(AoA)
        feats[20] = np.std(AoA)
    if len(IMG) > 0:
        feats[18] = np.mean(IMG)
        feats[21] = np.std(IMG)
    if len(FAM) > 0:
        feats[19] = np.mean(FAM)
        feats[22] = np.std(FAM)

    V,A,D = [],[],[]
    for word in words:
        if word in wDict:
            try:
                V.append(float(wDict[word][0]))
            except:
                pass
            try:
                A.append(float(wDict[word][1]))
            except:
                pass
            try:
                D.append(float(wDict[word][2]))
            except:
                pass
    if len(V) > 0:
        feats[23] = np.mean(V)
        feats[26] = np.std(V)
    if len(A) > 0:
        feats[24] = np.mean(A)
        feats[27] = np.std(A)
    if len(D) > 0:
        feats[25] = np.mean(D)
        feats[28] = np.std(D)
    return feats
def main( args ):

    catMap = {'Left':[0,lid,lfeat],'Center':[1,cid,cfeat],'Right':[2,rid,rfeat],'Alt':[3,aid,afeat]}
    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here
    for i in range(len(data)):
        feats[i] = extract1(data[i]['body'])
        feats[i][173] = catMap[data[i]['cat']][0]
        featIndex = catMap[data[i]['cat']][1].index(data[i]['id'])
        LIWC = catMap[data[i]['cat']][2][featIndex]
        feats[i][29:173] = LIWC
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 
    main(args)

