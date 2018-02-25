from __future__ import division, print_function

import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt

from sk import rdivDemo

import random

from collections import Counter

from mar import MAR
from wallace import Wallace



# from scipy.sparse import csr_matrix
# from sklearn.cluster import KMeans
#
# from time import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import svm
# import unicodedata
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# from sk import a12slow
# import csv



#### export

def export(file):
    read = MAR()
    read = read.create_lda(file)
    read.export_feature()

##### draw


def bestNworst(results):
    stats={}

    for key in results:

        if results[key]==[]:
            continue
        stats[key]={}
        result=results[key]
        order = np.argsort([r['x'][-1] for r in result])
        for ind in [0,25,50,75,100]:
            stats[key][ind]=result[order[int(ind*(len(order)-1)/100)]]

    return stats



##### UPDATE exp




## basic units

def START_DOC2VEC(filename):
    stop=0.95
    thres = 40

    read = MAR()
    read = read.create(filename)
    read.restart()
    read = MAR()
    read = read.create(filename)
    target = int(read.get_allpos()*stop)
    while True:
        pos, neg, total = read.get_numbers()
        print("%d, %d" %(pos,pos+neg))
        if pos >= target:
            break
        if pos==0 or pos+neg<thres:
            for id in read.random():
                read.code(id, read.body["label"][id])
        else:
            a,b,c,d,e =read.train(weighting=True)
            for id in c:
                read.code(id, read.body["label"][id])
    return read


###################################

##################


#############################



###
def draw_errors(seed=1):
    # np.random.seed(seed)
    # correct = BM25('Hall.csv', 'defect_prediction', 'est', 'random', 5).record
    # none = BM25('Hall.csv', 'defect_prediction', 'est', 'random').record
    # with open("../dump/error_30.pickle","a") as handle:
    #     pickle.dump(correct,handle)
    #     pickle.dump(none, handle)
    with open("../dump/error_30.pickle","r") as handle:
        correct = pickle.load(handle)
        none = pickle.load(handle)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)
    lines = ['-', '--', '-.', ':']
    five = ['$0th$', '$25th$', 'median result', '$75th$', 'worst result']
    colors = ["blue", 'red', 'green', 'brown', 'yellow']

    threelines = ['true_pos', 'false_neg', 'false_pos']
    names = {'true_pos':'true positives', 'false_neg':'false negatives', 'false_pos':'false positives'}

    plt.figure(0)
    for j, ind in enumerate(threelines):
        plt.plot(correct['count'], np.array(correct[ind]) , linestyle=lines[1], color=colors[j],label="Disagree (" + str(names[ind]) + ")")
    for j, ind in enumerate(threelines):
        plt.plot(none['count'], np.array(none[ind]) , linestyle=lines[0], color=colors[j],
                 label="None (" + str(names[ind]) + ")")
    # plt.ylabel("Recall")
    plt.xlabel("#Papers Reviewed")

    docnum = 8991
    x = [i * 100 for i in xrange(10)]



    xlabels = [str(z) + "\n(" + '%.1f' % (z / docnum * 100) + "%)" for z in x]
    plt.xticks(x, xlabels)

    # alldoc = 106
    # y = [i * 20 for i in xrange(6)]
    # ylabels = [str(z) + "\n(" + '%.2f' % (z / alldoc) + ")" for z in y]
    # plt.yticks(y, ylabels)

    plt.ylim((0, 100))
    plt.xlim((0, 900))

    plt.legend(bbox_to_anchor=(1, 0.65), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/error_all.eps")
    plt.savefig("../figure/error_all.png")



### BM25
def BM25(filename, query, stop='true', error='none', interval = 100000):
    stopat = 0.95
    thres = 0
    starting = 1
    interval = int(interval)


    read = MAR()
    read = read.create(filename)

    read.interval = int(interval)

    read.BM25(query.strip().split('_'))

    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    while True:
        pos, neg, total = read.get_error()
        # pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" % (pos, pos + neg))

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if (np.array(a)!=np.array(c)).any():
                if stop == 'est':
                    if stopat * read.est_num <= pos:
                        break
                elif stop == 'knee':
                    if pos>=10:
                        if read.knee():
                            if error=='random':
                                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                                seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                                part1 = set(seq[:read.kneepoint * read.step]) & set(
                                    np.where(np.array(read.body['code']) == "no")[0])
                                part2 = set(seq[read.kneepoint * read.step:]) & set(
                                    np.where(np.array(read.body['code']) == "yes")[0])
                                for id in part1|part2:
                                    read.code_error(id, error=error)
                            break
                else:
                    if pos >= target:
                        break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # if read.interval < 100:
    #     read.round = read.interval
    #     a,b,c,d =read.train(weighting=True,pne=True)
    #     for id in c:
    #         read.code_error(id, error=error)
    # read.export()
    results = analyze(read)
    print(results)
    return read

def analyze(read):
    unknown = np.where(np.array(read.body['code']) == "undetermined")[0]
    pos = np.where(np.array(read.body['code']) == "yes")[0]
    neg = np.where(np.array(read.body['code']) == "no")[0]
    yes = np.where(np.array(read.body['label']) == "yes")[0]
    no = np.where(np.array(read.body['label']) == "no")[0]
    falsepos = len(set(pos) & set(no))
    truepos = len(set(pos) & set(yes))
    falseneg = len(set(neg) & set(yes))
    unknownyes = len(set(unknown) & set(yes))
    unique = len(read.body['code']) - len(unknown)
    count = sum(read.body['count'])
    correction = read.correction
    return {"falsepos": falsepos, "truepos": truepos, "falseneg": falseneg, "unknownyes": unknownyes, "unique": unique, "count": count, "correction": correction}


############ scenarios ##########


def error_no_machine():
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction', "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "systematic review"}
    for file in files:
        print(file+": ", end='')
        BM25(file,queries[file],'est','three')

def error_machine():
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction', "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "systematic review"}
    for file in files:
        print(file+": ", end='')
        BM25(file,queries[file],'est','random')

def error_hpcc(seed = 1):
    # np.random.seed(int(seed))
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    queries = {"Hall.csv": 'defect_prediction', "Wahono.csv": 'defect_prediction', "Danijel.csv": 'defect_prediction_metrics', "K_all3.csv": "literature_review"}
    correct = ['none', 'three', 'machine', 'knee']

    results={}
    for file in files:
        results[file]={}
        for cor in correct:
            print(str(seed)+": "+file+": "+ cor+ ": ", end='')
            if cor == 'three':
                result = BM25(file,queries[file],'est','three')
            elif cor == 'machine':
                result = BM25(file,queries[file],'est','random', 5)
            elif cor == 'knee':
                result = BM25(file,queries[file],'knee','random')
            # elif cor == 'machine3':
            #     result = BM25(file,queries[file],'est','random3', 5)
            else:
                result = BM25(file,queries[file],'est','random')

            results[file][cor] = analyze(result)
    with open("../dump/error_new_hpcc00.pickle","a") as handle:
        pickle.dump(results,handle)

def error_summary():
    # import cPickle as pickle
    files = ["Hall.csv", "Wahono.csv", "Danijel.csv", "K_all3.csv"]
    correct = ['none', 'three', 'machine', 'knee']
    total = {"Hall.csv": 104, "Wahono.csv": 62, "Danijel.csv": 48, "K_all3.csv": 45}
    results = []
    with open("../dump/error_new_hpcc30.pickle","r") as handle:
        # result = pickle.load(handle)
        # result2 = pickle.load(handle)
        for i in xrange(18):
            results.append(pickle.load(handle))
            

        # results=[]
        # for i in xrange(30):
        #     results.append(pickle.load(handle))





    trans = {}

    for file in files:
        trans[file] = {}
        for cor in correct:
            trans[file][cor] = {}
            keys = results[0][file][cor].keys()
            for key in keys:
                trans[file][cor][key]=[]
                for i in xrange(len(results)):
                    trans[file][cor][key].append(results[i][file][cor][key])

    ####draw table

    print("\\begin{tabular}{ |l|"+"c|"*len(correct)+" }")
    print("\\hline")
    print("  & "+" & ".join(correct)+"  \\\\")
    print("\\hline")
    for dataset in files:
        # out = dataset.split('.')[0]+" & " + ' & '.join([str(int(np.median(trans[dataset][cor]['truepos'])))+" / "+ str(int(np.median(trans[dataset][cor]['count']))) +" / "+ str(int(np.median(trans[dataset][cor]['falseneg']))) +" / "+ str(int(np.median(trans[dataset][cor]['falsepos']))) for cor in correct]) + '\\\\'
        out = dataset.split('.')[0]+" & " + ' & '.join([str(round(np.median(np.array(trans[dataset][cor]['truepos'])/total[dataset]),2))+" / "+str(round(np.median(np.array(trans[dataset][cor]['truepos'])/(np.array(trans[dataset][cor]['truepos'])+np.array(trans[dataset][cor]['falsepos']))),2))+" / "+str(int(np.median(trans[dataset][cor]['count']))) for cor in correct]) + '\\\\'
        print(out)
        print("\\hline")
    print("\\end{tabular}")




if __name__ == "__main__":
    eval(cmd())
