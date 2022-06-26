import numpy as np
import pandas as pd 
from datetime import datetime
from matplotlib import pyplot as plt 
from mpl_toolkits.basemap import Basemap
from matplotlib import colors 
import seaborn as sns 


""" read in the city temperature data """ 

# reproducibility note:
# used "head -n 1 > Montreal.csv && grep -h Montreal Glob* >> Montreal.csv " due to dataset size 
df = pd.read_csv('Montreal.csv') 


""" take a quick look at the data  """ 

print(df.head()) # first ten lines

print(df.dtypes) # data types of each column 

print(df.shape) # shape of dataset

print(df.describe()) # some statistics 

""" some quick data processing """ 

# we do not care about rows where we have no temperature data 
df = df[~np.isnan(df['AverageTemperature'])]

def slice_to_year(date):
	return date[0:4]

df['dt'] = df['dt'].apply(str)
df['dt'] = df['dt'].apply(slice_to_year)
df['dt'] = df['dt'].apply(int)


df['AverageTemperature'] = df['AverageTemperature'].apply(float)
df['AverageTemperature'].dropna(how = 'any', axis = 0)

years = list(df['dt'])
temperatures = list(df['AverageTemperature'])

""" plot temperature in the city over time """ 
sns.lineplot(x = 'dt', y = 'AverageTemperature', data = df)

fitline = np.poly1d(np.polyfit(years, temperatures, 1)) # linear fit to temperature data 
# print(fitline)
plt.plot(years, fitline(years))
plt.xlabel('years')
plt.ylabel('temperature in C')
plt.title('Land Surface Temperature in Montreal')
plt.show()


""" how different are the regressions when we do a fifty year sliding window? """ 

yr_subsets = []
temp_subsets = []
chunk = 50 # looking at fifty year chunks 

for i in range(len(df['dt']) // chunk):
	
	yr_subset = years[chunk*i:chunk*2*i]
	temp_subset = temperatures[chunk*i:chunk*2*i]

	if yr_subset != []:
		yr_subsets.append(yr_subset)
	if temp_subset != []:
		temp_subsets.append(temp_subset)

# print(yr_subsets[0])
# plt.plot(yr_subsets[0], temp_subsets[0])

N = 50
fitline = np.poly1d(np.polyfit(yr_subsets[N], temp_subsets[N], 1)) # linear fit to temperature data 
# print(fitline)
plt.scatter(yr_subsets[N], temp_subsets[N], c = 'green')


plt.plot(yr_subsets[N], fitline(yr_subsets[N]))
plt.xlabel('years')
plt.ylabel('temperature in F')
plt.title('Land Surface Temperature in Montreal -- Subsetted')
plt.show()

year_changed = False 
current_year = years[0]
i = 0
temp_yr_avg = 0
avgtemps = []
ever = False
while i < len(years):
	
	count = 1 # keep track of how many entries have same year 
	if years[i] == current_year:
		count += 1
		temp_yr_avg += temperatures[i]
	else:
		current_year = years[i]
		temp_yr_avg = float(temp_yr_avg) / float(count)
		avgtemps.append(temp_yr_avg)
		temp_yr_avg = 0
		
	i += 1

print(avgtemps)
plt.title("annually averaged temperature")
plt.plot(list(set(years[2:len(years)])), avgtemps)
plt.show()

'''
eventually loop over script for many different cities
then plot a size and color coded temperature map weighted by certainty 


# creating map to plot data on 
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=N,resolution='c')


m = Basemap(llcrnrlon=-70,
            llcrnrlat=-60,
            urcrnrlon=260,
            urcrnrlat=60,
            lat_0=0,
            lon_0=180,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )

m.drawcoastlines()
'''

