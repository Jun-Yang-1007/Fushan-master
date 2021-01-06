#!/sur/bin/env python
#-*- coding: utf-8 -*-


import pyodbc

# server='172.27.61.100'
server='172.19.6.231'
port=1433

# base_database='prj_fj_env'    #业务数据，属于
# dts_database="prd_env_dts"   #监测数据 ，SENSORS  传送

base_database='PRD_WMS_2.0'    #业务数据，属于
dts_database="PRD_WMS_DTS_2.0"   #监测数据 ，SENSORS  传送

user='sa'
passwd='sa@123456'

# user='sa'
# passwd='fpi@123456'

def getConnection(server=None,database=None,user=None,passwd=None,driver='SQL Server',port=1433):
    connection = pyodbc.connect('DRIVER={'+driver+'};SERVER='+server+';PORT='+str(port)+ \
                                ';DATABASE='+database+';UID='+user+';PWD='+passwd+';TDS_Version=8.0')
    return connection


_base_connection = getConnection(server=server,database=base_database,user=user,passwd=passwd)
_dts_connection = getConnection(server=server,database=dts_database,user=user,passwd=passwd)



def getBaseConnection():

    return _base_connection

def getDtsConnection():
    return _dts_connection
