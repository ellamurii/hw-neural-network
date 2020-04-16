import numpy as np
from keras import engine
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import sys
from os import path
import json
fileName = sys.argv[1]

import metrics

def holt_winters(ts,slen,extra_periods=1, alpha=0.4, beta=0.4, phi=0.9,gamma=0.3):
    ts = ts.copy()
    try:
        ts = ts.tolist()
    except:
        pass
     
    def init_season(ts,slen):
        s = np.array([])
        ts = np.array(ts)
        for i in range(slen):
            col = [x for x in range(len(ts)) if x%slen==i]
                 
            s = np.append(s,np.mean(ts[col]))
         
        s /= np.mean(s)
        return s.tolist()
     
    s = init_season(ts,slen)
     
    f = [np.nan]
    a = [ts[0]/s[0]]
    b = [(ts[1]/s[1])-(ts[0]/s[0])]
     
    for t in range(1,slen):
        f.append((a[-1]+b[-1]*phi)*s[t])
        a.append(alpha*ts[t]/s[t]+(1-alpha)*(a[-1]+phi*b[-1]))
        b.append(beta*(a[-1]-a[-2])+(1-beta)*b[-1]*phi) 
     
    for t in range(slen,len(ts)):
        f.append((a[-1]+b[-1]*phi)*s[-slen])
        a.append(alpha*ts[t]/s[-slen]+(1-alpha)*(a[-1]+phi*b[-1]))
        b.append(beta*(a[-1]-a[-2])+(1-beta)*b[-1]*phi)
        s.append(gamma*ts[t]/a[-1] + (1-gamma)*s[-slen])

    for t in range(extra_periods):
        f.append((a[-1]+b[-1]*phi)*s[-slen])
        a.append(f[-1]/s[-slen])
        b.append(b[-1]*phi)
        s.append(s[-slen])
        ts.append(np.nan)
     
    dic = {"demand":ts,"forecast":f,"level":a,"trend":b,"season":s}
    results = pd.DataFrame.from_dict(dic)[["demand","forecast"]]
    results.index.name = "Period"
    results["error"] = results["demand"] - results["forecast"]

    if extra_periods: 
        print('\nNext Day Prediction:', round(f[-extra_periods:][0], 2))
    return results["forecast"]

basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "dataset", fileName))
dataset = pd.read_json(filepath)
data = dataset.iloc[:, 1]

slen = 7
pred_HW = holt_winters(data,slen)
smooth_value = holt_winters(data,slen,0)
metrics.evaluate(data,smooth_value)

plt.plot(pred_HW.index, pred_HW, label='Holt-Winters')
plt.plot(data.index, data, label='Actual Data')
plt.legend(loc='best')
plt.show()
