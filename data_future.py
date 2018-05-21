import pyodbc
import pandas as pd
import numpy as np

FILE = 1
def getConn():
    ip = '172.16.70.202'
    user = 'GTAUPDATE'
    pwd = 'GTAUPDATE'
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+ip+';UID='+user+';PWD='+pwd+';charset=utf8')
    return conn

def getFutureTickData():
    if FILE == 0:
        date = '20180502'
        sql = 'SELECT *  FROM [GTA_FFL2_TAQ_201805].[dbo].[FFL2_TAQ_IC1805_201805] where TDATE = \''+date+'\'' \
              ' and S1 > 0 order by TDATE,TTIME '
        data = pd.read_sql(sql,getConn())
        data.to_csv('data\\testData.csv')
    else:
        data = pd.read_csv('data\\testData.csv',encoding='gb2312')

    return data

def transposeFeature(data):
    windowLength = 20
    data_feature = data.loc[:, ['CP', 'CQ', 'CM', 'S1', 'S2', 'B1', 'B2', 'SV1', 'SV2', 'BV1', 'BV2', 'BSRATIO']].iloc[0:-windowLength + 1, :]
    feature = np.array(data_feature).tolist()
    data_label = data.loc[:,'CP'].rolling(windowLength).apply(lambda x:x[-1]/x[0]-1)[windowLength-1:]
    label = np.array(data_label).tolist()
    return feature,label


if __name__ == '__main__':
    tickData = getFutureTickData()
    trainData = transposeFeature(tickData)
    print('ok')
