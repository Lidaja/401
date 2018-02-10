import sys
import argparse
import os
import json
import re
import string
import spacy
import html

indir = '/u/cs401/A1/data/';
nlp = spacy.load('en',disable=['parser','ner'])

def remove_extra_space(comment):
    return re.sub(r"(  +)",r" ",comment.rstrip().lstrip())

def remove_newline(comment):
    return remove_extra_space(comment.replace("\n",""))

def replace_html(comment):
    return remove_extra_space(" ".join([html.unescape(c) for c in comment.split(" ")]))

def remove_urls(comment):
    return remove_extra_space(re.sub(r'http[^\s]*|www[^\s]*','',comment))

def split_punctuation(comment):
    puncList = list(string.punctuation)
    puncList.remove("'")
    puncList.remove(".")
    puncs = "\\"+"\\".join(puncList)
    puncRegex = r"(\w)(["+ puncs+r"]+)"
    comment = re.sub(puncRegex, r"\1 \2 ", comment)
    comment = re.sub(r"(\w)(\.+["+puncs+"]+)",r"\1 \2 ",comment)
    comment = re.sub(r"(\w{2,})\.(\w*)", r"\1 . \2", comment)
    comment = re.sub(r"(\w*)\.(\w{2,})", r"\1 . \2", comment)
    return remove_extra_space(comment)

def split_clitics(comment):
    return remove_extra_space(re.sub(r"(\w)(\w'[^s]\s|'s|s'|'\w+)",r"\1 \2",comment))

def spacy_tag(comment):
    if not comment:
        return comment
    doc = spacy.tokens.Doc(nlp.vocab, words=comment.split(" "))
    doc = nlp.tagger(doc)
    tokens = []
    for token in doc:
        tokens.append(token.text+"/"+token.tag_)
    return remove_extra_space(" ".join(tokens))

def remove_stopwords(comment):
    comments = comment.split(" ")
    StopWords = [word.rstrip() for word in open('/u/cs401/Wordlists/StopWords','r').readlines()]
    for t in range(len(comments)-1,-1,-1):
        token = comments[t].rsplit("/",1)[0]
        if token.lower() in StopWords:
            comments.pop(t)
    return remove_extra_space(" ".join(comments))

def lemmatize(comment):
    if not comment:
        return comment
    commentSplit = [word.rsplit("/",1)[0] for word in comment.split(" ")]
    doc = spacy.tokens.Doc(nlp.vocab, words = commentSplit)
    doc = nlp.tagger(doc)
    tokens = []
    for token in doc:
        tokens.append(token.lemma_+"/"+token.tag_)
    return remove_extra_space(" ".join(tokens))

def add_newline(comment):
    abbrev = [word.rstrip() for word in open('/u/cs401/Wordlists/abbrev.english','r').readlines()]
    commentSplit = comment.split(" ")
    for c in range(len(commentSplit)-1,-1,-1):
        token = commentSplit[c].rsplit("/",1)[0]
        if token == ".":
            if c > 0:
                if commentSplit[c-1].rsplit("/",1)[0]+"." not in abbrev:
                    commentSplit.insert(c+1,"\n")
    return remove_extra_space(" ".join(commentSplit))

def make_lowercase(comment):
    comments = comment.split(" ")
    for c in range(len(comments)):
        commentSplit = comments[c].rsplit("/",1)
        if commentSplit[0] != "\n":
            comments[c] = commentSplit[0].lower() + "/" + commentSplit[-1]
    return remove_extra_space(" ".join(comments))

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    #print("before=",comment)
    modComm = ''

    if 1 in steps:
        comment=remove_newline(comment)
        #print("Step1:",comment)
    if 2 in steps:
        comment=replace_html(comment)
        #print("Step2:",comment)
    if 3 in steps:
        comment = remove_urls(comment)
        #print("Step3:",comment)
    if 4 in steps:
        comment = split_punctuation(comment)
        #print("Step4:",comment)
    if 5 in steps:
        comment = split_clitics(comment)
        #print("Step5:",comment)
    if 6 in steps:
        comment = spacy_tag(comment)
        #print("Step6:",comment)
    if 7 in steps:
        comment = remove_stopwords(comment)
        #print("Step7:",comment)
    if 8 in steps:
        comment = lemmatize(comment)
        #print("Step8:",comment)
    if 9 in steps:
        comment = add_newline(comment)
        #print("Step9:",comment)
    if 10 in steps:
        comment = make_lowercase(comment)
        #print("Step10:",comment)
    modComm = comment
    #print("after=",modComm)
    return modComm

def main( args ):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            cutData = data[:args.max]
            for line in cutData:

                # TODO: read those lines with something like `j = json.loads(line)`
                j = json.loads(line)
                # TODO: choose to retain fields from those lines that are relevant to you
                for key in list(j.keys()):
                    if key not in ['ups','downs','score','controversiality','subreddit','author','body','id']:
                        j.pop(key,None)
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                j[u'cat'] = fullFile.split('/')[-1]
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                procResult = preproc1(j['body'])
                # TODO: replace the 'body' field with the processed text
                j['body'] = procResult
                allOutput.append(j)
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()
    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    main(args)
