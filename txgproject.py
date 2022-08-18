#!/usr/bin/env python
# coding: utf-8

# ## Preparation

# In[ ]:


# Install packages if not available 
get_ipython().run_line_magic('pip', 'install kaggle --upgrade')


# In[ ]:


'''
Prepare packages
'''
# Table manipulation
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Kaggle API
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


# ## Import Data

# In[ ]:


'''
Method to Import Datasets:
    - Use Kaggle API to import directly from data source
    - Unzip file and extract to /dbfs/FileStore/Tables in Databricks
'''
def load_to_DBFS(df,
                df_new):
    '''
    Load to Databricks DBFS
    '''
    # File location and type
    file_location = f"/FileStore/tables/{df}"
    file_type = "csv"

    # CSV options
    infer_schema = "false"
    first_row_is_header = "true"
    delimiter = ","

    # The applied options are for CSV files. For other file types, these will be ignored.
    df = spark.read.format(file_type)       .option("inferSchema", infer_schema)       .option("header", first_row_is_header)       .option("sep", delimiter)       .load(file_location)
    df.write.format("parquet").mode("overwrite").saveAsTable(f"{df_new}")

def import_from_kaggle():
    global mp_daily, mp_world, mp_timeline
    
    api = KaggleApi()
    api.authenticate()
    
    # Optional: search for Kaggle directory with monkeypox data, change the string inside single quotes to any other datasets that you want
    # !kaggle datasets list -s 'monkeypox'
    
    get_ipython().system("kaggle datasets download -d 'deepcontractor/monkeypox-dataset-daily-updated' --force")
    get_ipython().system('unzip -o monkeypox-dataset-daily-updated.zip -d /dbfs/FileStore/tables')
    
    # Spark format
    Daily_Country_Wise_Confirmed_Cases = spark.read.csv('/FileStore/tables/Daily_Country_Wise_Confirmed_Cases.csv', header = True)
    Monkey_Pox_Cases_Worldwide = spark.read.csv('/FileStore/tables/Monkey_Pox_Cases_Worldwide.csv', header = True)
    Worldwide_Case_Detection_Timeline = spark.read.csv('/FileStore/tables/Worldwide_Case_Detection_Timeline.csv', header = True)
    
    # Load to Databricks DBFS, prepare for exporting to other applications if required e.g.: Power BI
    load_to_DBFS(df="Daily_Country_Wise_Confirmed_Cases.csv", df_new="Daily_Country_Wise_Confirmed_Cases")
    load_to_DBFS(df="Monkey_Pox_Cases_Worldwide.csv", df_new="Monkey_Pox_Cases_Worldwide")
    load_to_DBFS(df="Worldwide_Case_Detection_Timeline.csv", df_new="Worldwide_Case_Detection_Timeline")
    
    # Convert to Pandas dataframe for easy manipulation
    mp_daily = Daily_Country_Wise_Confirmed_Cases.toPandas()
    mp_world = Monkey_Pox_Cases_Worldwide.toPandas()
    mp_timeline = Worldwide_Case_Detection_Timeline.toPandas()


# In[ ]:


import_from_kaggle()


# ## Clean and Modify Datasets

# In[ ]:


'''
Clean data
'''

# Clean mp_daily dataset
daily_country = mp_daily.T
daily_country["date"] = mp_daily.T.index
daily_country.reset_index(inplace=True, drop=True)
daily_country.columns = [daily_country.iloc[0]]
daily_country.drop(index=0, inplace=True) # drop the first row
daily_country.reset_index(inplace=True, drop=True)
daily_country.rename(columns = {'Country':'Date'}, inplace = True)
daily_country.dtypes

col_name = list(mp_daily["Country"])
daily_country[col_name] = daily_country[col_name].apply(pd.to_numeric)

daily_date = daily_country["Date"]
temp = daily_country.drop("Date", axis=1)
row_sum = temp.sum(axis = 1)
row_sum
temp["daily_total"] = row_sum
temp["Date"] = daily_date
mp_daily_t = temp

col_list = list(mp_daily.columns)[1:len(list(mp_daily.columns))]
for i in col_list:
    mp_daily[i] = pd.to_numeric(mp_daily[i])


# In[ ]:


# Clean mp_world dataset
num_list = ["Confirmed_Cases",
           "Suspected_Cases",
           "Hospitalized"]
for i in num_list:
    mp_world[i] = pd.to_numeric(mp_world[i])


# In[ ]:


dcwc = mp_daily.copy() #daily count of confirmed monkeypox cases by country
mpcw = mp_world.copy() #total number of confirmed, suspected, and hospitalized cases by country
wcdt = mp_timeline.copy() #age, gender, sympotoms, hospitalization, isolation, and travel history for patients (separated by country and city)


# ## Preliminary Analysis

# In[ ]:


inf = mpcw.nlargest(columns = 'Confirmed_Cases', n = 10)
target = inf['Country'].tolist()
dcwc.loc['Total', :] = dcwc.sum(axis = 0)

bigs = dcwc[(dcwc['Country'] == target[0]) | 
            (dcwc['Country'] == target[1]) |
            (dcwc['Country'] == target[2]) |
            (dcwc['Country'] == target[3]) |
            (dcwc['Country'] == target[4]) |
            (dcwc['Country'] == target[5]) |
            (dcwc['Country'] == target[6]) |
            (dcwc['Country'] == target[7]) |
            (dcwc['Country'] == target[8]) |
            (dcwc['Country'] == target[9])]

get_ipython().run_line_magic('matplotlib', 'inline')
work = bigs.set_index('Country')
trial = work.transpose()
trial.plot(title = 'Daily Cases by Country (Top 10 Countries)',
          xlabel = 'Date',
          ylabel = 'Confirmed_Cases')


# In[ ]:


bigs = mpcw[(mpcw['Country'] == target[0]) | 
            (mpcw['Country'] == target[1]) |
            (mpcw['Country'] == target[2]) |
            (mpcw['Country'] == target[3]) |
            (mpcw['Country'] == target[4]) |
            (mpcw['Country'] == target[5]) |
            (mpcw['Country'] == target[6]) |
            (mpcw['Country'] == target[7]) |
            (mpcw['Country'] == target[8]) |
            (mpcw['Country'] == target[9])]

plt.bar(x = bigs['Country'], height = bigs['Confirmed_Cases'], color = ['red', 'purple', 'green', 'blue', 'orange'])
plt.xticks(rotation = 90)
plt.title('Total Confirmed Cases By Country (Top Ten Countries)')
plt.xlabel('Country')
plt.ylabel('Confirmed Cases Total')
plt.show()


# In[ ]:


# Clean mp_timeline
mp_timeline.Symptoms = mp_timeline.Symptoms.str.lower()


# In[ ]:


inf = mpcw.nlargest(columns = 'Suspected_Cases', n = 10)
target = inf['Country'].tolist()
dcwc.loc['Total', :] = dcwc.sum(axis = 0)

bigs = mpcw[(mpcw['Country'] == target[0]) | 
            (mpcw['Country'] == target[1]) |
            (mpcw['Country'] == target[2]) |
            (mpcw['Country'] == target[3]) |
            (mpcw['Country'] == target[4]) |
            (mpcw['Country'] == target[5]) |
            (mpcw['Country'] == target[6]) |
            (mpcw['Country'] == target[7]) |
            (mpcw['Country'] == target[8]) |
            (mpcw['Country'] == target[9])]

plt.bar(x = bigs['Country'], height = bigs['Suspected_Cases'], color = ['red', 'purple', 'green', 'blue', 'orange'])
plt.xticks(rotation = 90)
plt.title('Total Suspected Cases By Country (Top Ten Countries)')
plt.xlabel('Country')
plt.ylabel('Suspected Cases Total')
plt.show()


# In[ ]:


# Top symptoms by count worldwide
temp = mp_timeline
temp = temp.sort_values(by = ['Country', 'City', 'Date_confirmation'])

symptoms_list = temp.Symptoms.str.split(", ")
symptoms_list = filter(None, symptoms_list)
symptoms_df = pd.DataFrame(symptoms_list) 

col_list=[]
for i in range(0,len(symptoms_df.columns),1):
    temp_col = list(symptoms_df[i])
    col_list = col_list + temp_col
    
col_list = filter(None, col_list)
symptoms_df = pd.DataFrame(col_list)
symptoms_df.rename(columns = {0:'Symptom'}, inplace = True)
symptoms_gp = symptoms_df.groupby('Symptom').size().reset_index(name='Count').sort_values(['Count'], ascending = False).head(10).reset_index().drop("index", axis = 1)

sns.set_theme(style="whitegrid")
#tips = sns.load_dataset("tips")
ax = sns.barplot(x="Count", y="Symptom", data=symptoms_gp,
                palette="Blues_d")


# In[ ]:


# Hospitalized cases by country
sns.set_style('whitegrid')
fig,axes = plt.subplots(figsize=(20,8))
# order = mp_timeline.groupby('Country').size().sort_values()[mp_timeline.groupby('Country').size().sort_values() > 1].index[::-1]
order = mp_timeline.groupby('Country').size().sort_values().index[::-1]
ax = sns.countplot(x="Country", data=mp_timeline, hue = "Hospitalised (Y/N/NA)",order=order)
for container in ax.containers:
    ax.bar_label(container)
plt.legend(title="Hospitalised (Y/N)", bbox_to_anchor=(1.15,1), loc='upper right', borderaxespad=0.)
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.title("Hospitalised Cases By Country")
plt.show()


# In[ ]:


# Travel history by country
sns.set_style('whitegrid')
fig,axes = plt.subplots(figsize=(20,8))
order = mp_timeline.groupby('Country').size().sort_values().index[::-1]
ax = sns.countplot(x="Country", data=mp_timeline, hue = "Travel_history (Y/N/NA)",order=order)
for container in ax.containers:
    ax.bar_label(container)
plt.legend(title="Travel_history (Y/N/NA)", bbox_to_anchor=(1.15,1), loc='upper right', borderaxespad=0.)
plt.xticks(rotation = 90)
plt.title("Travel History By Country")
plt.show()


# In[ ]:


# Confirmed, suspected, hospitalized cases over world map
def world_map(df, col=None,title=None):
    '''
    Function to plot a choropleth world Map.
    Arguments required:
    1. Column Name for which distribution is to be plotted.
    2. The Title of the Graph.
    '''
    fig = px.choropleth(df,
                  locations='Country',
                  locationmode='country names',
                  hover_name='Country',
                  color=col,
                  color_continuous_scale='blues')

    fig.update_layout(title_text=title)
    fig.show()
world_map(df = mp_world, col = 'Confirmed_Cases', title = "Confirmed Cases by Country")
world_map(df = mp_world, col = 'Suspected_Cases', title = "Suspected Cases by Country")
#world_map(df = mp_world, col = 'Hospitalized', title = "Hospitalized Cases by Country")

