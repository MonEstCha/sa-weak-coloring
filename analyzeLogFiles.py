#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:33:28 2024

@author: monique
Execution example: 
    import analyzeLogFiles as al
    al.wcolVsClusterCoeff('/path/to/directory')
"""
import os
import matplotlib.pyplot as plt 
import numpy as np
import networkx as nx
import math
from scipy.stats import pearsonr
from functools import reduce  # forward compatibility for Python 3
import operator

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

##
# Recurses the passed datastructure and returns the minimal weak coloring number
##
def cmpRecDict(data, level):
    if level > 0:
        result = {'wcol': -1, 'ct': 1, 'runtime': 0}
    else:
        result = {}

    for dkey in data.keys():
        # at lowest level?
        if not isinstance(data[dkey], dict):
            return data
        # at rad level
        elif level == 0:
            result[dkey] = cmpRecDict(data[dkey], level + 1)
        else:
            keyRes = cmpRecDict(data[dkey], level + 1)
            wcolAvg = keyRes['wcol'] / keyRes['ct']
            lower = (result['wcol'] / result['ct']) > wcolAvg
            if result['wcol'] == -1 or lower:
                result = keyRes
            #res = cmpRecDict(data[dkey])
    return result
            

##
# Function determines the average value of the passed property and its standard deviation if stdev is True
##
def getAvgValuePerRadAndGraph(data, prop, stdev):
    allGraphsRes = {}
    for n, key in enumerate(data):
        #print(key)
        result = 0
        graphData = data[key]['v1']
        minResult = cmpRecDict(graphData, 0)
        for radKey in minResult.keys():
            if radKey not in allGraphsRes.keys():
                allGraphsRes[radKey] = {}
            obj = minResult[radKey]
            if obj['ct'] > 0:
                result = obj[prop]['sum'] / obj['ct'] 
                if stdev:
                    allGraphsRes[radKey][key] = {prop: result, 'stdev': obj[prop]['stdev']}
                else:
                    allGraphsRes[radKey][key] = result

    return allGraphsRes
    
##
# Function plots the runtimes for the log files in the passed directory
# @param lf_dir, the logfile directory
# @param graph_dir, the graphfile directory
##
def plotRunTimes(lf_dir, graph_dir):

    data = readInDatatree_MF(['version', 'Radius', 'heuristic', 'swaps'], [0,1,2], ['wcol', 'runtime'], [4,5], "#Radius", '   0.00', lf_dir)
    stats, names = getStats(graph_dir)
    avgRTperGraph = getAvgValuePerRadAndGraph(data, 'runtime', False)
    #for radKey in avgRTperRadAndGraph.keys():
     #   for graph in avgRTperRadAndGraph.keys():
        
    pltRowCt = math.ceil(len(avgRTperGraph.keys())/2)
    pltColCt = 2
    fig, axs = plt.subplots(pltRowCt, pltColCt, figsize=(10, 15))

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.9)
    for radKey in avgRTperGraph.keys():
        # init datastructures for current radius
        runtimes = []
        e_v_sums = []
        radObj = avgRTperGraph[radKey]
        for graphKey in radObj:
            graphName = graphKey.split('_simAnneal')[0]
            runtimes.append(radObj[graphKey])
            _sum = stats[graphName]['nodes'] + stats[graphName]['edges']
            e_v_sums.append(_sum)
            
            
        _rad = int(radKey) - 1
        x_coord = math.floor(_rad/2)
        y_coord = _rad%2

        _e_v_sums, _runtimes = sortLists(e_v_sums, runtimes)
        corr, _ = pearsonr(_e_v_sums, _runtimes)
        axs[x_coord, y_coord].scatter(_e_v_sums, _runtimes, marker='o',color='k', s=10, label = "Correlation: %.3f" % corr)
            
        trend_n = np.polyfit(_e_v_sums, _runtimes, 1)
        p = np.poly1d(trend_n)
        axs[x_coord, y_coord].set_yscale("log")
        axs[x_coord, y_coord].set_xscale("log")
        axs[x_coord, y_coord].legend(fontsize=7) 
        axs[x_coord, y_coord].plot(_e_v_sums, p(_e_v_sums),"k--")
        axs[x_coord, y_coord].set_title('Radius ' + radKey)
        
        fig.tight_layout()
    
    ct = 0
    for ax in axs.flat:
        if ct%2 == 0:
            ax.set_ylabel('Runtime (s)')
        if ct > 5:
            ax.set_xlabel('|E| + |V|')
        ct += 1


##
# Function compares two versions of Simulated Annealing
# @param lf_dir, the logfile directory
# @param selection, the graph subset to be analyzed (if empty, all logfiles in lf_dir are analyzed)
# @param landscape, the plot format
# @param labels, the identifiers of the versions
##
def compareApproachesPerRadius(lf_dir, selection, landscape, labels):
    selEmpty = len(selection) == 0
    radArray = ['1', '2', '3', '4', '5', '6', '7', '8']
    #labels = {'v1': 'exponential', 'v0': 'logarithmic'}
    labels = {'v1': 'swapRandomly', 'v2': 'reducedSearchSpace'}
    end = '1.4'
    result_heur = readHeuristicData_AllLogFiles(lf_dir)
    data = readInDatatree_MF(['version', 'Radius'], [0,1,2], ['wcol', 'runtime'], [4,5], "#Radius", '   0.00', lf_dir)
    logfiles = data.keys()
       
    plotCt = len(logfiles) if selEmpty else len(selection)

    if landscape:
        #landscape for 3 cols
        pltColCt = 3
        plt.figure(figsize=(11, 5))
    else:
        plt.figure(figsize=(10, 12))
        pltColCt = 2
    
    pltRowCt = math.ceil(plotCt/pltColCt)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    #fig.tight_layout(h_pad=2)
    #print(pltRowCt, pltColCt)
    n = 0
    for key in logfiles:
        # add a new subplot iteratively
        graphName = key.split('_simAnneal_v1')[0]
        if not selEmpty and graphName not in selection:
            continue
        ax = plt.subplot(pltRowCt, pltColCt, n + 1)
        ax.set_title(graphName)
        SAObj = data[key]
                
        heurObj = result_heur[key]
        wreach, sreach, flatw, sortd, deg = [], [], [], [], []
        if 'wreach' in heurObj.keys():
            wreach = heurObj['wreach']
        if 'sreach' in heurObj.keys():
            sreach = heurObj['sreach']
        if 'flatw' in heurObj.keys():
            flatw = heurObj['flatw']
        if 'sortd' in heurObj.keys():
            sortd = heurObj['sortd']
        #if 'deg' in heurObj.keys():
         #   deg = heurObj['deg']
                    
        # add heuristic plots
        if wreach: 
            ax.plot(radArray, wreach, marker='o', color = 'r', label = "ByWreachLeft", markersize=5)
        if sreach:
            ax.plot(radArray, sreach, marker='o', color = 'g', label = "BySreachRight", markersize=5)
        if flatw:
            ax.plot(radArray[0:len(flatw)], flatw, marker='o', color = 'blue', label = "FlatWcol", markersize=5)
        if sortd:
            ax.plot(radArray[0:len(sortd)], sortd, marker='o', color = 'orange', label = "SortDeg", markersize=5)
        #if deg:
         #   ax.plot(radArray[0:len(deg)], deg, marker='o', color = 'violet', label = "Degeneracy", markersize=5)
            
        # add SA plots
        rads = []
        wcols = []
        length = len(SAObj.keys())
        ct = length-1
        offset = 1.5 / length*1.0
        weights = np.arange(0.,0.4*length, offset)

        for vKey in SAObj.keys():
            versionObj = SAObj[vKey]
            for rad in versionObj.keys():
                _data = versionObj[rad]['data']
                _data = getFromDict(_data, ['0.006', '0.2', end])
                if _data['ct'] != 0 :
                    rads.append(rad)
                    wcols.append(_data['wcol']['sum']/_data['ct'])
            # a plot for each swapNr
            labelStr = labels[vKey]
            ax.plot(rads, wcols, marker='o', color = (weights[ct]/2.0,weights[ct]/2.0, weights[ct]/2.0), label = labelStr, markersize=5)
            ct -= 1 
            rads = []
            wcols = []
            
        ax.legend(fontsize=5)
        if n%pltColCt == 0:
            ax.set_ylabel('Wcol')
        if n > plotCt - (pltColCt + 1):
            ax.set_xlabel('Radius')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        n += 1
        
    
##
# Function plots the average weak coloring numbers per radius
# @param lf_dir, the logfile directory
# @param selection, the graph subset to be analyzed (if empty, all logfiles in lf_dir are analyzed)
# @param initOrder, the heuristic used for the start vertex order
##
def wcolVsRadius(lf_dir, initOrder, selection):
    selEmpty = len(selection) == 0
    radArray = ['1', '2', '3', '4', '5', '6', '7', '8']
    result_SA = readSAData_AllLogFiles(lf_dir)
    result_heur = readHeuristicData_AllLogFiles(lf_dir)
    plotCt = len(result_SA.keys()) if selEmpty else len(selection)
    print('plotct: ' + str(plotCt))
    pltColCt = 3
    pltRowCt = math.ceil(plotCt/pltColCt)
    #plt.figure(figsize=(10, 15))
    plt.figure(figsize=(11, 5))
    n = 0
    
    for key in result_SA.keys():
        graphName = key.split('_simAnneal')[0]
        if not selEmpty and graphName not in selection:
            continue
        # add a new subplot iteratively
        ax = plt.subplot(pltRowCt, pltColCt, n + 1)
        ax.set_title(graphName, fontsize=10)
        #print(n, key)
        #print("Extracting data for " + graphName)
        SAObj = result_SA[key]
        heurObj = {}
        prefix = graphName[0:4]
        # if prefix == "path" or prefix == "star":
        #     SAObj = result_SA[key]['random']
        # else:
        #     SAObj = result_SA[key]['none'] 
            # get an array with the average value of each wcol per swap and radius

        avgWcolsPerSwap = getAverageWcolPerInitOrder(SAObj)
        if initOrder in avgWcolsPerSwap.keys():
            avgWcolsPerSwap = avgWcolsPerSwap[initOrder]
        elif 'none' in avgWcolsPerSwap.keys():
            avgWcolsPerSwap = avgWcolsPerSwap['none']
        
        heurObj = result_heur[key]
        wreach, sreach, flatw, sortd, deg = [], [], [], [], []
        if 'wreach' in heurObj.keys():
            wreach = heurObj['wreach']
        if 'sreach' in heurObj.keys():
            sreach = heurObj['sreach']
        if 'flatw' in heurObj.keys():
            flatw = heurObj['flatw']
        if 'sortd' in heurObj.keys():
            sortd = heurObj['sortd']
        #if 'deg' in heurObj.keys():
         #   deg = heurObj['deg']
                   
        # add heuristic plots
        if wreach: 
            ax.plot(radArray, wreach, marker='o', color = 'r', label = "ByWreachLeft", markersize=5)
        if sreach:
            ax.plot(radArray, sreach, marker='o', color = 'g', label = "BySreachRight", markersize=5)
        if flatw:
            ax.plot(radArray[0:len(flatw)], flatw, marker='o', color = 'blue', label = "FlatWcol", markersize=5)
        if sortd:
            ax.plot(radArray[0:len(sortd)], sortd, marker='o', color = 'orange', label = "SortDeg", markersize=5)
        #if deg:
         #   ax.plot(radArray[0:len(deg)], deg, marker='o', color = 'violet', label = "Degeneracy", markersize=5)
            
        # add SA plots
        rads = []
        wcols = []
        length = len(avgWcolsPerSwap.keys())
        ct = length-1
        offset = 1.5 / length*1.0
        weights = np.arange(0.,0.4*length, offset)

        for swapKey in avgWcolsPerSwap.keys():
            swapObj = avgWcolsPerSwap[swapKey]
            
            for _key, _value in swapObj.items():
                if _value != -1 :
                    rads.append(_key)
                    wcols.append(_value)
            # a plot for each swapNr
            labelStr = swapKey + ' swaps'
            ax.plot(rads, wcols, marker='o', color = (weights[ct]/2.0,weights[ct]/2.0, weights[ct]/2.0), label = labelStr, markersize=5)
            ct -= 1 
            rads = []
            wcols = []
            
        # finalize plot
        ax.legend(fontsize=5, ncol=2)
        if n%pltColCt == 0:
            ax.set_ylabel('Wcol')
        if n > plotCt - (pltColCt + 1):
            ax.set_xlabel('Radius')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        n += 1
        

##
# Function calculates the deviation, i.e. the ratio of the best achieved weak coloring number by SA 
# to that achieved by the best heuristic
# @param lf_dir, the logfile directory
# @return the deviations per radius per graph
##
def getDeviations(lf_dir):
    print("Reading in data...")
    result_SA = readSAData_AllLogFiles(lf_dir)
    result_heur = readHeuristicData_AllLogFiles(lf_dir)
    devsAllGraphsPerRad = {}
    print("Done")
    print("Extracting data...")
    for key in result_SA.keys():
        graphName = key.split('_simAnneal')[0]
        #print("Extracting data for " + graphName)
        SAObj = {}
        heurObj = {}
        SAObj = result_SA[key]
        heurObj = result_heur[key]
        # get an array with the average value of each wcol per swap and radius
        avgWcolsPerSwap = getAverageWcolPerInitOrder(SAObj)
        # for each radius determine relative difference of best wcol at any swap to best heuristic
        deviation = getDeviation(avgWcolsPerSwap, heurObj)
        # add each relative value to corresponding array attribute of result obj
        for rad in deviation.keys():
            if rad not in devsAllGraphsPerRad.keys():
                devsAllGraphsPerRad[rad] = {graphName: deviation[rad]}
            else:
                devsAllGraphsPerRad[rad][graphName] = deviation[rad]
                
    print("Done") 
    return devsAllGraphsPerRad

def plotValuePerStat_ls(devAllGraphsPerRad, stats, names, ccs_gephi, talkVersion):  
    # retrieve the keys of any stats entry

    statsKeys = list(stats.values().__iter__().__next__().keys())

    for skey in statsKeys:
        plotCt = 6 if talkVersion else math.ceil(len(devAllGraphsPerRad.keys()))
        print('plotct: ' + str(plotCt))
        pltColCt = 3 if talkVersion else 2
        pltRowCt = math.ceil(plotCt/pltColCt)
        if not talkVersion:
            plt.figure(figsize=(10, 15))
        else:
            plt.figure(figsize=(22, 10))
        n = 0
    
        for rad in sorted(devAllGraphsPerRad.keys()):
            if talkVersion:
                if rad == '4' or rad == '6':
                    continue
            ax = plt.subplot(pltRowCt, pltColCt, n + 1)
            
            dev = []
            attr = []
            if skey == 'cc':
                ccs_g = []
            for graphName in devAllGraphsPerRad[rad].keys():
                curDev = devAllGraphsPerRad[rad][graphName]
                if graphName in stats.keys() and curDev != '-':           
                    dev.append(curDev)
                    if skey == 'cc':
                        if stats[graphName][skey] != '-':
                            attr.append(stats[graphName][skey])
                        else:
                            attr.append(ccs_gephi[graphName])
                        ccs_g.append(ccs_gephi[graphName])
                    else:
                        attr.append(stats[graphName][skey])
                        
            if len(dev) == 0:
                continue

            _attr, _dev = [], []
            if skey == 'cc':
                _ccs_g, _dev = sortLists(ccs_g, dev)
                corr, _ = pearsonr(_ccs_g, _dev)
                ax.scatter(_ccs_g, _dev, marker='+', color='grey', s=20)
                z = np.polyfit(_ccs_g, _dev, 2)
                p = np.poly1d(z)
                ax.plot(_ccs_g, p(_ccs_g),"k--", label = 'Fitting function: %.3f' % p[0] + '*CC\u00b2 + %.3f' % p[1] + '*CC + %.3f' % p[2])

            else:
                _attr, _dev = sortLists(attr, dev)
                corr, _ = pearsonr(_attr, _dev)
                ax.scatter(_attr, _dev, marker='o',color='grey', s=10)

            
            ax.set_title('Radius ' + rad + ", correlation: %.3f" % corr)
            
            if talkVersion:
                ax.legend(fontsize=10) 
            else:
                ax.legend(fontsize=5) 
        
            if n%pltColCt == 0:
                ax.set_ylabel('Deviation')
            if n > plotCt - (pltColCt + 1):
                ax.set_xlabel(names[skey])
            else:
                ax.xaxis.set_tick_params(labelbottom=False)
            n += 1



def plotValuePerStat(devAllGraphsPerRad, stats, names, ccs_gephi):  
    # retrieve the keys of any stats entry

    statsKeys = list(stats.values().__iter__().__next__().keys())

    for skey in statsKeys:
        pltRowCt = math.ceil(len(devAllGraphsPerRad.keys())/2)
        pltColCt = 2
        fig, axs = plt.subplots(pltRowCt, pltColCt, figsize=(10, 15))

        plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.6, 
                        hspace=0.9)
    
        for rad in sorted(devAllGraphsPerRad.keys()):
            _rad = int(rad) - 1
            x_coord = math.floor(_rad/2)
            y_coord = _rad%2
            dev = []
            attr = []
            if skey == 'cc':
                ccs_g = []
            for graphName in devAllGraphsPerRad[rad].keys():
                curDev = devAllGraphsPerRad[rad][graphName]
                if graphName in stats.keys() and curDev != '-':           
                    dev.append(curDev)
                    if skey == 'cc':
                        if stats[graphName][skey] != '-':
                            attr.append(stats[graphName][skey])
                        else:
                            attr.append(ccs_gephi[graphName])
                        ccs_g.append(ccs_gephi[graphName])
                    else:
                        attr.append(stats[graphName][skey])
                        
            if len(dev) == 0:
                continue
           # if skey == 'cc':
            #    ccs_g = [a for _, a in sorted(zip(dev, ccs_g))]
            #attr = [a for _, a in sorted(zip(dev, attr))]
            #dev = dev.sort()

            _attr, _dev = [], []
            if skey == 'cc':
                _ccs_g, _dev = sortLists(ccs_g, dev)
                corr, _ = pearsonr(_ccs_g, _dev)
                axs[x_coord, y_coord].scatter(_ccs_g, _dev, marker='+', color='grey', s=20)
                z = np.polyfit(_ccs_g, _dev, 2)
                p = np.poly1d(z)
                axs[x_coord, y_coord].plot(_ccs_g, p(_ccs_g),"k--", label = 'Fitting function: %.3f' % p[0] + '*CC\u00b2 + %.3f' % p[1] + '*CC + %.3f' % p[2])

            else:
                _attr, _dev = sortLists(attr, dev)
                corr, _ = pearsonr(_attr, _dev)
                axs[x_coord, y_coord].scatter(_attr, _dev, marker='o',color='grey', s=10)
                if skey != 'avgDeg' and skey != 'maxDeg':
                    trend_n = np.polyfit(_attr, _dev, 2)
                    p = np.poly1d(trend_n)
                    axs[x_coord, y_coord].plot(_attr, p(_attr),"k--", label = 'Fitting function: %.3f' % p[0] + '*CC\u00b2 + %.3f' % p[1] + '*CC + %.3f' % p[2])
                
            axs[x_coord, y_coord].set_title('Radius ' + rad + ", correlation: %.3f" % corr)
                
            axs[x_coord, y_coord].legend(fontsize=7) 
            
            fig.tight_layout()
        
        ct = 0
        for ax in axs.flat:
            if ct%2 == 0:
                ax.set_ylabel('Deviation')
            if ct > 5:
                ax.set_xlabel(names[skey])
            ct += 1



def plotDevVsStats(lf_dir, graph_dir, talkVersion):  
    ccs_gephi = {'ego-facebook': 0.801, 'iprob': 0.00, 'mk10-b4': 0.00, 'gams60am': 0.006, 'rajat19': 0.437,
                 'dwt_1005': 0.456,'bcsstk20': 0.514,'small':0.001,'bcspwr06':0.072, 'fs_183_1': 0.51, 'filter2D': 0.421, 
                 'pores_2': 0.279, 'bio-diseasome': 0.77, 'ch4-4-b2': 0.04, 'poisson2D': 0.438, 'plsk1919': 0.421,       
                'curtis54': 0.559,'delaunay_n10': 0.436, 'fs_680_3': 0.591, 'fpga_trans_02': 0.573,
                 'petster-friendships-hamster': 0.167, 'de063155': 0, 'oscil_dcop_01': 0.191,
                 'NotreDame_yeast': 0.153, 'diseasome': 0.819, 'bio-grid-plant': 0.274, 'celegans': 0.308,
                 'yeast': 0.20, 'bio-yeast-protein-inter': 0.171, 'bio-yeast': 0.14, 'bio-grid-fission-yeast': 0.312,
                 'ba_7000_2_2': 0.0036703008270927324, 'ba_1500_4_2': 0.006188653889429276, 'star_40_50': 0.0,
                 'path_3000': 0.0, 'ba_500_9_8': 0.08, 'ba_3000_2_1': 0.0, 'ba_1500_9_4': 0.016, 'ba_200_20_20': 0.273,
                 'ba_3000_7_4__2': 0.012, 'ba_300_12_8': 0.111, 'ba_3000_7_4': 0.013, 'path-6019': 0.0, 'ba_600_3_5perc': 0.062,
                   'ba_600_3_40perc': 0.252, 'ba_600_3': 0.303, 'ba_600_3_nr2': 0.543}

    devAllGraphsPerRad = getDeviations(lf_dir)
    stats, names = getStats(graph_dir)
    
    #plotValuePerStat(devAllGraphsPerRad, stats, names, ccs_gephi)
    plotValuePerStat_ls(devAllGraphsPerRad, stats, names, ccs_gephi, talkVersion)
    
    
        
##
# Sorts a bunch of list dependent on the sort order of the first one
# @return the sorted lists
##                
def sortLists(*args):
    leadList = args[0]
    indexSorted = sorted(range(len(leadList)), key=lambda k: leadList[k])
    args_sorted = [[cur[k] for k in indexSorted] for cur in args]
   
    return args_sorted
    

def printStats(G):
    edges = dict(G.edges()).values()
    nodeNr = len(G)
    degrees = dict(G.degree()).values()
    avg = sum(degrees)/float(nodeNr)
    _max = max(degrees)
    print('average degree: ' + str(avg))
    print('maximum degree: ' + str(_max))
    print('nodeNr: ' + str(nodeNr))
    print('edges: ' + str(len(edges)))


##
# Function iterates the passed graph directory and reads in the statistics for each graph
# @param graph_dir, the directory to scan
# @return the statistics per graph
##    
def getStats(graph_dir):
    print("Calculating statistical data ...")
    stats = {}
    
    def extFilter(x):
        if x.split('.')[1] in ['csv', 'txtg', 'edges']:
          return True
        else:
          return False

    for root, subdirs, files in os.walk(graph_dir):
        files = filter(extFilter, files)
        for name in files:
            graphName = name.split('.')[0]
            g = nx.Graph()
            fullName = os.path.join(root, name)
            el = []
            nodes = []
            edges = []
            #print("Processing " + fullName)
            
            with open(fullName, 'r') as filebuf:
                for row in filebuf:
                    if row.startswith('%') or row.startswith('Target') or row.startswith('Source') or row.startswith('#'):
                        continue
                    if ',' in row:
                        el = row.split(',')
                    elif ';' in row:
                        el = row.split(';')
                    elif '    ' in row:
                        el = row.split('    ') 
                    elif ' ' in row:
                        el = row.split(' ')  
                    # omit self-loops as we want g to be undirected
                    if el[0] != el[1]:
                        nodes.append(int(el[0]))
                        nodes.append(int(el[1]))
                        edges.append([int(el[0]), int(el[1])])
                   
            nodes = list(set(nodes)) 
            edges = list(set(tuple(sorted(sub)) for sub in edges))
            #print(edges)
            
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            nodeNr = len(g)
            edgeCt = len(dict(g.edges()).values())
                
            degrees = dict(g.degree()).values()
            avg = sum(degrees)/float(nodeNr)
            _max = max(degrees)
            #print(graphName)
            
            stats[graphName] = {'nodes': nodeNr, 'edges': edgeCt, 'avgDeg': avg, 'maxDeg': _max}
    print("Done")
    names = {'nodes': '|V|', 'edges': '|E|', 'avgDeg': r'$\bar{d}$', 'maxDeg': u'Δ(G)'}
                
    return stats, names
            
        
def getDeviation(avgWcols, heurObj):
    # for all radii get the lowest weak coloring nr achieved by any heuristic
    heurMin = {'1': -1,'2': -1,'3': -1,'4': -1,'5': -1,'6': -1,'7': -1,'8': -1}

    for heur in heurObj.keys():
        ct = 1
        for el in heurObj[heur]:
            if heurMin[str(ct)] == -1:
                heurMin[str(ct)] = el
            else:
                heurMin[str(ct)] = min(el, heurMin[str(ct)])
            ct += 1
        
    # for all radii get the lowest weak coloring nr achieved for any initial order and at any swap number by the SA algorithm          
    wcolMin = {'1': -1,'2': -1,'3': -1,'4': -1,'5': -1,'6': -1,'7': -1,'8': -1}
    for initKey in avgWcols.keys():
        initObj = avgWcols[initKey]
        for swaps in initObj.keys():
            swapObj = initObj[swaps]
            for radius in swapObj.keys():
                if wcolMin[radius] == -1:
                    wcolMin[radius] = swapObj[radius]
                else:
                    wcolMin[radius] = min(swapObj[radius], wcolMin[radius])
    
    deviation = {'1': '-','2': '-','3': '-','4': '-','5': '-','6': '-','7': '-','8': '-'}
    for key in heurMin.keys():
        if not(heurMin[key] == -1 or wcolMin[key] == -1):
            deviation[key] = wcolMin[key]/heurMin[key]

    return deviation  

##
# Function identifies the minimal weak coloring number achieved for a graph for any setting
# and the standard deviation
#
def getMinWcolStats(lf_dir):
    data = readInDatatree_MF(['Radius', 'heuristic', 'swaps'], [0,1,2], ['wcol', 'runtime'], [4,5], "#Radius", '   0.00', lf_dir)
   
    avgWcolAndStdevPerGraph = getAvgValuePerRadAndGraph(data, 'wcol', True)
    
    print(avgWcolAndStdevPerGraph)
    

##
# Function calculates the average weak coloring number of a graph per initial order, swap nr. and radius
# @param SAObj the sum and number of executions of identical SA experiments for a given graph per initial order, swap nr. and radius, i.e.
# in the form {initKey: {swapKey: {radKey: {wcol: value, ct: value}, ...}}}
# @return an object storing the average weak coloring number of a graph per initial order, swap nr. and radius   
##
def getAverageWcolPerInitOrder(SAObj):
    allWcolAvgs = {}
    for initKey in SAObj.keys():
        initObj = SAObj[initKey]
        allWcolAvgs[initKey] = {}
        for swapKey in initObj.keys():
            wcolAvg = {'1': -1,'2': -1,'3': -1,'4': -1,'5': -1,'6': -1,'7': -1,'8': -1}
            swapObj = initObj[swapKey]
            for radKey in swapObj.keys():
                radObj = swapObj[radKey]            
                if radObj['ct'] != 0:
                    wcolAvg[radKey] = radObj['wcol']/radObj['ct']
            allWcolAvgs[initKey][swapKey] = wcolAvg
    return allWcolAvgs   
        

def readSAData_AllLogFiles(lf_dir):
    #get all log files in the working directory
    # in case log files reside in a different directory
    allResults = {}
    result = {}
    
    def extFilter(x):
        if x.split('.')[1] in ['txt']:
          return True
        else:
          return False

    for root, subdirs, files in os.walk(lf_dir):
        #print(subdirs)
        files = filter(extFilter, files)
        for file in files:
            #print(subdirs,root, file)
            #print('Processing ' +file)
            logName = file.split('.txt')[0]
            file = root + '/' + file
            #infile = file
            if file == lf_dir + "/celegans_simAnneal_1704194983.txt":
                result = readInSAData('rad', 'wcol', '0.006', '-', '0.2', '2.4', file)
            else:
                result = readInSAData('rad', 'wcol', '0.006', '-', '0.2', '1.4', file)
            allResults[logName] = result
    
    return allResults

def readHeuristicData_AllLogFiles(lf_dir):
    #get all log files in the working directory
    # in case log files reside in a different directory
    allResults = {}
    result = {}
    
    def extFilter(x):
        if x.split('.')[1] in ['txt']:
          return True
        else:
          return False

    for root, subdirs, files in os.walk(lf_dir):
        files = filter(extFilter, files)
        for file in files:
            #print('Processing ' +file)
            logName = file.split('.txt')[0]
            file = root + '/' + file
            result = readInHeuristicData(file)
            allResults[logName] = result
    
    return allResults

##
# Reads in the data of multiple files as specified by the order of levels
# @param configLevels, the tree levels w.r.t. params in the config lines
# @param innerLevelPos, the positions of params used as tree levels, that are in the actual data lines
# @param dataKeys, the data to read in e.g. wcol or runtime
# @param dataPositions, at which position the actual data will be found in a line - necessary since not preceded by an id
# @param configStart, the beginning of the config lines
# @param dataStart, the beginning of the data lines
# @param infile, the logfile to process 
# @return a datatree holding the asked data structured in the specified way
##
def readInDatatree_MF(configLevels, innerLevelPos, dataKeys, dataPositions, configStart, dataStart, lf_dir):
    #get all log files in lf_dir
    allResults = {}
    result = {}
    
    def extFilter(x):
        if x.split('.')[1] in ['txt']:
          return True
        else:
          return False

    for root, subdirs, files in os.walk(lf_dir):
        #print(subdirs)
        files = filter(extFilter, files)
        for file in files:
            #print(subdirs,root, file)
            #print('Processing ' +file)
            logName = file.split('.txt')[0]
            file = root + '/' + file
            #infile = file
            result = readInDatatree_SF(configLevels, innerLevelPos, dataKeys, dataPositions, configStart, dataStart, file)
            allResults[logName] = result
    
    return allResults
    

##
# Reads in the data from the passed file as specified by the order of levels
# @param configLevels, the tree levels w.r.t. params in the config lines
# @param innerLevelPos, the positions of params used as tree levels, that are in the actual data lines
# @param dataKeys, the data to read in e.g. wcol or runtime
# @param dataPositions, at which position the actual data will be found in a line - necessary since not preceded by an id
# @param configStart, the beginning of the config lines
# @param dataStart, the beginning of the data lines
# @param infile, the logfile to process 
# @return a datatree holding the asked data structured in the specified way
##
def readInDatatree_SF(configLevels, innerLevelPos, dataKeys, dataPositions, configStart, dataStart, infile):
    print(infile)
    results = {}
    innerDict = {}
    ct = 0
    with open(infile, 'r') as filebuf:     
        for row in filebuf: 
            #crnt = results
            if row.startswith(configStart):
                crnt = results
                for el in configLevels:
                    param = row.split(el + ': ')[1].split(', ')[0].split('\n')[0] 
                    # follow one level deeper
                    crnt = crnt.setdefault(param, {})  
                crnt['data'] = []
            if row.startswith(dataStart):
                crnt['data'].append(row)
    results = get_inner_dict(results, len(configLevels) + 1, dataPositions, dataKeys, innerLevelPos) 
    
    return results

##
# Function converts a bunch of rows into a dict using the parameters in the row as keys for the actual data
# Attention: Only works if data entries at innerLevelPos are the same for all rows
##
def get_inner_dict(results, depth, dataPos, dataKeys, innerLevelPos):
    print(results)
    _dict = {}
    if depth == 0:
        ct = 0       
        _dictInner = {}
        values = {}
        for row in results:
            params = row.split(',   ')
            crnt = _dict
            for ipos in innerLevelPos[:-1]:
                param = params[ipos].split('   ')[-1].split('\n')[0]
                crnt = crnt.setdefault(param, {})
            ct += 1
            innerCt = 0
            for pos in dataPos:
                if dataKeys[innerCt] not in _dictInner.keys():
                    _dictInner[dataKeys[innerCt]] = {'sum': 0, 'stdev': 0}
                if dataKeys[innerCt] not in values.keys():
                    values[dataKeys[innerCt]] = []
                # remove any unwanted spaces and linebreaks from the value
                param = params[pos].split('   ')[-1].split('\n')[0]
                # sum up over all rows
                _dictInner[dataKeys[innerCt]]['sum'] += float(param)
                # store single values to calculate standard deviation of data
                values[dataKeys[innerCt]].append(float(param))
                innerCt += 1
        print(values)
        for i in range(2):
            _dictInner[dataKeys[i]]['stdev'] = np.std(values[dataKeys[i]])
        _dictInner['ct'] = ct
        param = params[innerLevelPos[-1]].split('   ')[-1].split('\n')[0]
        crnt.setdefault(param, _dictInner)
        return _dict
    
    else:
        for key in results.keys():
            #print('key: ' + key)
            _dict[key] = {}
            _dict[key] = get_inner_dict(results[key], depth-1, dataPos, dataKeys, innerLevelPos)
        return _dict
                
        
def readInSAData(type_x, type_y, slope, _swapNr, _startT, _endT, infile):
    results = {}
    bySwaps = type_x != 'startT'
    with open(infile, 'r') as filebuf:
        for row in filebuf: 
            if row.startswith('#Radius'):
                params = row.split(': ')
                rad = params[1][0]
                swapNr = params[3][0:2].split('\n')[0]
                heur = params[2][0:-7]
                
                if (not bySwaps and swapNr == _swapNr):
                    if not rad in results.keys():
                        results[rad] = {}
                        
                elif bySwaps:
                    if not heur in results.keys():
                        results[heur] = {}
                    if not swapNr in results[heur].keys():
                        results[heur][swapNr] = {}
                    if not rad in results[heur][swapNr].keys():
                        results[heur][swapNr][rad] = {}
                        results[heur][swapNr][rad] = {'wcol': 0, 'ct': 0, 'runtime': 0}
                                                      
            elif row.startswith('   ' + slope):
                array = row.split(',   ')
                startT = array[1]
                endT = array[2]
                if bySwaps and startT == _startT and endT == _endT: 
                    results[heur][swapNr][rad]['wcol'] += int(array[4]) 
                    results[heur][swapNr][rad]['ct'] += 1
                elif (not bySwaps and swapNr == _swapNr):
                    if not startT in results[rad].keys():
                        results[rad][startT] = {}
                    if not endT in results[rad][startT].keys():
                        results[rad][startT][endT] = {}
                        results[rad][startT][endT] = {'wcol': 0, 'ct': 0, 'runtime': 0}
                    results[rad][startT][endT]['wcol'] += int(array[4]) 
                    results[rad][startT][endT]['runtime'] += int(array[3]) 
                    results[rad][startT][endT]['ct'] += 1
                    
        return results

def readInHeuristicData(infile):
    wreach = []
    sreach = []
    flatw = []
    sortd = []
    deg = []
    lowerBound = 0
    result = {}
    with open(infile, 'r') as filebuf:
        for row in filebuf: 
            if row.startswith('#wReachLeft'):
                reachSt = row.split(':')[1][0:-1].split(',')
                for el in reachSt:
                    wreach.append(int(el))
                result['wreach'] = wreach
                lowerBound = max(wreach)
            elif row.startswith('#sReachRight'):
                reachSt = row.split(':')[1][0:-1].split(',')
                for el in reachSt:
                    sreach.append(int(el))
                result['sreach'] = sreach
                lowerBound = max(lowerBound, max(sreach))
            elif row.startswith('#FlatWcol'):
                reachSt = row.split(':')[1][0:-1].split(',')
                for el in reachSt:
                    if int(el) <= lowerBound:
                        flatw.append(int(el))
                result['flatw'] = flatw
            elif row.startswith('#SortDeg'):
                reachSt = row.split(':')[1][0:-1].split(',')
                for el in reachSt:
                    if int(el) <= lowerBound:
                        sortd.append(int(el))
                result['sortd'] = sortd
            elif row.startswith('#Degeneracy'):
                reachSt = row.split(':')[1][0:-1].split(',')
                for el in reachSt:
                    if int(el) <= lowerBound:
                        deg.append(int(el))
                result['deg'] = deg
    return result
