#!/sur/bin/env python
#-*- coding: utf-8 -*-

from .connection import getBaseConnection

CODE_WSYSTEM='2c938239477b7c8f01477ba676ed0002'
CODE_WSYSTEM='2c90827271c3757f0171c4b757790005'
MONITOR_TYPE="WMS"

VALLEY_TABLE="BASE_VALLEY"
FACTOR_TABLE='BASE_DATAITEM'
SITE_TABLE='WMS_WATER_SITE'

connection = getBaseConnection()

def getSites():

    cursor = connection.cursor()

    sites=[]

    sql="SELECT S.ID ,S.POINTNAME FROM {0} S WHERE CODE_WSYSTEM IN (".format(SITE_TABLE) + \
        "SELECT V.ID FROM {0} V WHERE V.ID = '{1}' OR V.PARENTID = '{1}')".format(VALLEY_TABLE,CODE_WSYSTEM)

    cursor.execute(sql)

    while 1:
        row = cursor.fetchone()
        if not row:
            break
        sites.append({'id':row.ID,'name':row.POINTNAME})

    cursor.close()

    return  sites

def getFactors():

    cursor = connection.cursor()

    factors=[]

    # water monitor factors

    SQL="SELECT ID,COLUMNCODE,PRECISION FROM {0} WHERE MONITORTYPE='{1}' and  COLUMNCODE in ('NH3','TP','TN')".format(FACTOR_TABLE,MONITOR_TYPE)

    cursor.execute(SQL)

    while 1:
        row = cursor.fetchone()
        if not row:
            break
        factors.append({
            "id":row.ID,
            "code":row.COLUMNCODE,
            "column":"F_"+row.COLUMNCODE
        })

    cursor.close()

    return factors
 

if __name__=='__main__':

    # query sites.

    sites=getSites()
    print("Sites list -----------------------------------")
    for site in sites:
        print(site)

    factors = getFactors()

    print("Factors list ---------------------------------")
    for factor in factors:
        print(factor)    
