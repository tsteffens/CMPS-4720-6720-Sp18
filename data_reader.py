import pandas as pd

#read data from file and split by ,
asos = pd.read_csv('asos.txt', index_col = False, sep=",")

#chop off the remarks
asos['metar'] = asos['metar'].map(lambda x:x.split("RMK",1)[0])

#split metar column by " "
metar1= asos['metar'].str.split(' ',expand =True)

#drop metar column from asos
asos1 = asos.drop(asos.columns[-1], axis = 1)
metar1 = metar1.drop(metar1.columns[2], axis =1)

#combine the metar back into asos
asos1 = pd.concat([asos1, metar1], axis =1)


#write data to csv file seperated by #
asos1.to_csv("test.csv", sep = '#', index = False)
