# -*- coding: utf-8 -*-

import urllib2
import time
import datetime
import pickle
import sched

def getLeaderboard():
    response = urllib2.urlopen('https://competitions.codalab.org/competitions/11711/results/21941/data')
    html = response.read()
        
    try:
        entries = pickle.load(open('leaderboard.pkl','rb'))
    except:
        entries = {}
    
    for line in html.split('\n')[1:]:
        line = line.split(',')
        if len(line) == 4:
            line = line[:3]
            line[2] = line[2].split(' ')[0]

            currentScore = float(line[2])
            teamname = line[1]
            savedEntry = entries.get(teamname, [35.0, None, -1])
            savedScore = savedEntry[0]
            currentCount = savedEntry[2]
            
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')    
            
            if currentScore < savedScore:
                entries[teamname] = [currentScore, st, currentCount + 1]

    with open('leaderboard.pkl','wb') as f:
        pickle.dump(entries,f)

def printLeaderboard():
    entries = pickle.load(open('leaderboard.pkl','rb'))
    for i, entry in enumerate(sorted(entries.items(), key=lambda e: e[1][0])):
        print i+1, '\t', entry[1][0], '\t', entry[1][1], '\t', entry[1][2], '\t', entry[0]
        
#%%
s = sched.scheduler(time.time, time.sleep)
def do_something(sc): 
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')   
    print ''
    print '====== CURRENT LEADERBOARD DSG 2016 (', st, ') ======'
    getLeaderboard()
    printLeaderboard()
    s.enter(60, 1, do_something, (sc,))

s.enter(1, 1, do_something, (s,))
s.run()
        
