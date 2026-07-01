#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:39:14 2026

@author: mdraminski
"""
# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/

import pandas as pd
import numpy as np
import statistics
import boto3
import os
import json
import logging
import sys
import snowflake.connector as sf
import sfutils as sutil
import utils as util
#from openpyxl as exl
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from ollama import chat
from embeddings import embeddings

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

logger = logging.getLogger(__name__)
util.configure_logging()


###################################
###################################
###################################
conn_params = {
    'account': 'SFDC_DP_PRD',
    'user': 'SVC_BT_BU_MLI_DEVELOPER_PRD',
    'authenticator': 'SNOWFLAKE_JWT',
    'private_key': util.get_private_key_from_secrets_manager(),
    'warehouse': 'WH_MLI_ETL',
    'database': 'SSE_DM_MKT_BROKEN',
    'schema': 'WRK_TRUTH_PROFILE'
}
ctx = sf.connect(**conn_params)
cs = ctx.cursor()

#########################
data_query = {
              "events":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_EVENT;",
              "sessions":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION;",
              "speakers":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION_SPEAKER WHERE SPEAKER_ID is not NULL;",
              "attendee":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_ATTENDEE WHERE ATTENDEE_ID is not NULL;",
              "session_room":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_ROOM_TIME;",
              #"rooms":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_ROOMS;",
              "meetings":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_MEETING;",              
              "attendance":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_ATTENDANCE;",
              "bookmarks":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION_BOOKMARKS;",              
              "favorites":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_FAVORITES;",
              }


#SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_EVENT
#SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION_SPEAKER 
#SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION_BOOKMARKS
#SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_ROOMS
#SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_ATTENDEE
#SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_ATTENDANCE
#SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_FAVORITES

#SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_ROOM_TIME
#SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SURVEY_RESPONSES
#SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_MEETING

#wtcopenhagen26
#"SELECT * FROM SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION Where EVENTCODE='wtcopenhagen26'"


#drop_columns_all = ['FILES','FILE_NAME','FILE_ROW_NUMBER']
drop_columns_all = ['FILES', 'FILE_NAME','FILE_ROW_NUMBER','FILE_LAST_MODIFIED','AUDIT_ETL_JOB_INS_TS','AUDIT_ETL_JOB_UPD_TS']

data_drop_columns = {
              "events":drop_columns_all,
              "sessions":drop_columns_all,
              "speakers":drop_columns_all,
              "attendee":drop_columns_all,
              "session_room":drop_columns_all,
              #"rooms":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_ROOMS;",
              "meetings":drop_columns_all,
              "attendance":drop_columns_all,
              "bookmarks":drop_columns_all,              
              "favorites":drop_columns_all,
              }

data_focus_columns = {
              "events":['EVENTCODE', 'NAME','DISPLAY_NAME','CITY','TYPE','START_DATE','END_DATE','COUNTRY_ID','STATE_ID','STATUS','VENUE_NAME','DESCRIPTION'],
              "sessions":['SESSION_ID', 'SESSIONCODE', 'TITLE', 'ABSTRACT', 'LENGTH', 'EVENTCODE', 'EVENT_NAME','EVENT_ID','TIMES_OFFERED','SESSION_TYPE','SESSION_TRACK','SESSION_SUMMARY_BY_EINSTEIN', 'KEY_TAKEAWAYS_BY_EINSTEIN', 'SESSION_MEDIUM'],
              "speakers":["EVENTCODE","SESSION_ID","SPEAKER_ID","ATTENDEE_ID", "GLOBAL_FULL_NAME","GLOBAL_COMPANY","GLOBAL_JOB_TITLE","BIO"],
              "attendee":['EVENTCODE', 'ATTENDEE_ID', 'STATUS', 'JOB_TITLE', 'FIRST_NAME', 'LAST_NAME', 'EMAIL', 'COMPANY_NAME', 'ATTENDEE_TYPE','CANCELLED','TEST_RECORD','APPROVAL_STATUS','REGISTERED','CHECKIN_DATE','PRIMARY_ROLE','PRIMARY_INDUSTRY'],
              "session_room":['EVENTCODE', 'SESSION_ID', 'ROOM_ID', 'ROOM', 'START_TIME', 'END_TIME', 'DAY_NAME', 'LENGTH', 'CAPACITY', 'UTC_START_TIME', 'UTC_END_TIME'],
              #"rooms":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_ROOMS;",
              "meetings":['EVENTCODE', 'ATTENDEE_ID', 'SESSION_ID', 'CODE', 'PARTICIPANT_COMPANY', 'PARTICIPANT_EMAIL', 'PARTICIPANT_FIRST_NAME', 'PARTICIPANT_LAST_NAME', 'PARTICIPANT_ROLES', 'TITLE', 'ABSTRACT', 'STATUS', 'MEETING_PROGRAM_ID', 'MEETING_PROGRAM_NAME', 'ROOM', 'MEETING_TOPIC', 'MEETING_DESCRIPTION', 'START_TIME_UTC', 'END_TIME_UTC'],
              "attendance":['EVENTCODE', 'CODE', 'SESSION_ID', 'ATTENDEE_ID', 'ATTENDEDDATE','ATTENDEDTIME', 'UTCATTENDEDTIME', 'ATTENDANCE_STATUS'],
              "bookmarks":['EVENTCODE', 'EMAIL', 'SESSION_ID', 'ATTENDEE_ID', 'SESSION_TIME', 'SESSION_DATE', 'SESSION_TITLE', 'SESSION_CODE', 'SESSION_TYPE', 'FIRST_NAME', 'LAST_NAME', 'SCHEDULED_DATETIME', 'ATTENDED_DATETIME'],
              "favorites":[],
              }

data_path = '/Users/mdraminski/Workspace/SessionFlow/data/'
tmp_path = '/Users/mdraminski/TEMP4/'
results_path = '/Users/mdraminski/Workspace/SessionFlow/Results/' 
attendee_mli_email = ['rhandt@salesforce.com', 'hchipman@salesforce.com','peter.gee@salesforce.com','dhamerla@salesforce.com']


######################
### FETCH THE DATA ###
for dt in data_query:
    logger.info("Fetching dataset: %s", dt)
    df = util.fetch_pandas_all(cs, data_query[dt])
    df = df.drop(columns=data_drop_columns[dt], errors='ignore')
    if dt == 'sessions':
        sutil.session_filter(df, True).to_csv(data_path + dt + "_filtered.csv", index = False)
    df.to_csv(data_path + dt + ".csv", index = False)




#########################
### LOAD AND PREPARE THE DATA ###
######################
events = pd.read_csv(data_path + "events.csv")
sessions = pd.read_csv(data_path + "sessions.csv")
sessions = sutil.session_filter(sessions, True, version = 2)
sessions.shape
speakers = pd.read_csv(data_path + "speakers.csv")
#session_room = pd.read_csv(data_path + "session_room.csv")

### Calculate SESSIONS Embeddings
session_embeddings = sutil.calc_session_embeddings(sessions, speakers)
logger.info("Embeddings shape: %s", session_embeddings.shape)
type(session_embeddings)
session_embeddings = embeddings(session_embeddings, sessions['SESSION_ID'])
session_embeddings.shape()

### Calculate Embedding Similarities (session_sims)
model = SentenceTransformer("all-MiniLM-L6-v2")
session_sims = model.similarity(session_embeddings.embeddings, session_embeddings.embeddings).numpy()
len(session_sims)
logger.info("Similarity matrix shape: %s", session_sims.shape)

session_sims = pd.DataFrame(session_sims, index=sessions['SESSION_ID'], columns=sessions['SESSION_ID'])
session_sims.index.name = session_sims.columns.name= None
logger.info("Similarity session_sims matrix:\n%s", session_sims.shape)


### LOAD ATTENDEE
attendee = pd.read_csv(data_path + "attendee.csv")
attendee = attendee.loc[attendee['TEST_RECORD'] != 'Yes']
attendee = attendee[data_focus_columns['attendee']]
len(attendee)

attendee_mli = attendee.loc[attendee['EMAIL'].isin(attendee_mli_email), ['ATTENDEE_ID','EMAIL']]
attendee_mli = attendee_mli.drop_duplicates()


### LOAD ATTENDANCE
attendance = pd.read_csv(data_path + "attendance.csv")
#attendance[:3000].to_csv('~/attendance3000.csv')
attendance = attendance[data_focus_columns['attendance']]
len(attendance)
#only legit sessions
attendance = attendance.loc[attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))]
len(attendance)

### LOAD bookmarks
bookmarks = pd.read_csv(data_path + "bookmarks.csv")
# add ATTENDEE_ID to bookmarks
bookmarks = bookmarks.merge(attendee[['ATTENDEE_ID','EMAIL']].drop_duplicates(), how = 'left', on = 'EMAIL')
bookmarks = bookmarks[~bookmarks['ATTENDEE_ID'].isnull()]
bookmarks = bookmarks[data_focus_columns['bookmarks']]
#bookmarks[:1000].to_csv("~/bookmarks.csv")
# SOME SESSIONS MAY occur many times on one conference
bookmarks.loc[bookmarks['SESSION_ID']=='1722538248635001Pgq8',['EVENTCODE','SESSION_ID','SESSION_TIME','SESSION_DATE', 'SESSION_TITLE']].drop_duplicates().sort_values(by=['SESSION_DATE','SESSION_TIME'])
bookmarks = bookmarks[bookmarks['SESSION_ID'].isin(sessions['SESSION_ID'])]
bookmarks[['EVENTCODE','SESSION_ID','ATTENDEE_ID']]
len(bookmarks)



######################
# THE EXPERIMENT
eventcode = 'df25'
#eventcode = 'tdx26'
######################
recs_params = {
    "ret_recs_size": 30,
    "user_nn_size": 25,    
    "popular_weight": 0.3,
    "nn_weight": 1,
    "emb_weight": 1,
}


# File path
results_file_path = results_path + eventcode + "_result.xlsx"
sheet_name = "Sheet1"
#writer.close()

writer = None
ws = None
#writer = pd.ExcelWriter(results_file_path, engine="openpyxl")
#ws = writer.sheets[sheet_name]

calc_recs = False
#calc_recs = True

add_mli_users = True

### TABLE ** EVENTS **  ###
_event_end_dates = events.loc[events['EVENTCODE'].isin([eventcode]),'END_DATE']
if _event_end_dates.empty:
    raise ValueError(f"No event found for eventcode={eventcode!r}")
event_date = _event_end_dates.iloc[0]
past_events = events.loc[events['END_DATE'] < event_date,'EVENTCODE']
#events.loc[events['EVENTCODE'].isin(past_events)].to_csv('~/events.csv')
event_row = events.loc[events['EVENTCODE']==eventcode, data_focus_columns['events']]
# Write to a specific position
if writer is not None:
    event_row[['EVENTCODE','NAME','CITY','START_DATE','END_DATE','VENUE_NAME']].to_excel(writer,sheet_name=sheet_name,startrow=0,startcol=0,index=False)
    main_title = event_row['NAME'].iloc[0] + ' ' + event_row['CITY'].iloc[0] + '(' + event_row['START_DATE'].iloc[0] + ') SessionFlow Recommendation Results'    
    ws.cell(row=3, column=1).value = main_title


### TABLE ** SESSIONS **  ###
event_sessions = sessions.loc[sessions['EVENTCODE'].isin([eventcode])].reset_index()
len(event_sessions)
#event_sessions.to_csv('~/event_sessions.csv')
event_sessions['SESSION_TYPE'].value_counts()


### TABLE ** SESSION_ROOM **  ###
session_room = pd.read_csv(data_path + "session_room.csv")
session_room = session_room[data_focus_columns['session_room']]
session_room["START_TIME"] = (pd.to_datetime(session_room["UTC_START_TIME"], utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
session_room["END_TIME"] = (pd.to_datetime(session_room["UTC_END_TIME"], utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
session_room = session_room.drop(columns=["UTC_START_TIME", "UTC_END_TIME"])
session_room = session_room.loc[session_room['SESSION_ID'].isin(event_sessions['SESSION_ID'])]
pd.DataFrame(session_room['START_TIME'].value_counts()).reset_index().sort_values('START_TIME').to_csv('~/session_room_times.csv')
session_room.sort_values(by=["START_TIME"]).merge(sessions[['SESSION_ID','TITLE']], how='left',on='SESSION_ID').to_csv('~/session_room.csv')


### TABLE ** ATTENDEE ** ###
event_attendee = attendee.loc[attendee['EVENTCODE']==eventcode, data_focus_columns['attendee']]
#event_attendee[:1000].to_csv('~/attendee.csv')
len(event_attendee) #22788

#ADD MLI USERS
if add_mli_users:
    event_attendee = pd.concat([event_attendee, attendee.loc[attendee['ATTENDEE_ID'].isin(attendee_mli['ATTENDEE_ID']),event_attendee.columns].groupby(["ATTENDEE_ID"]).last().reset_index()])
    event_attendee['EVENTCODE'] = eventcode

#ADD attendee description
cols = ["FIRST_NAME", "LAST_NAME", "COMPANY_NAME","JOB_TITLE", "PRIMARY_ROLE", "PRIMARY_INDUSTRY"]
event_attendee[cols] = event_attendee[cols].fillna("")
event_attendee["description"] = event_attendee.apply(sutil.build_attendee_description, axis=1)
event_attendee["description"] = (event_attendee["description"].fillna("").astype(str).str.strip())
event_attendee['description'].iloc[1]
event_attendee.shape
#event_attendee.to_csv('~/real_attendee.csv')

#BUILD user_embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
user_embeddings = model.encode(event_attendee['description'].to_list())
user_embeddings = embeddings(user_embeddings, event_attendee['ATTENDEE_ID'])


#Count event attendee status
user_sessions_filter = sutil.get_sessions_filter(event_attendee, event_sessions)
user_sessions_filter = None

event_attendee_cnt = pd.DataFrame(event_attendee['STATUS'].value_counts()).reset_index()
event_attendee_cnt.columns = ['STATUS','Attendees']
event_attendee_cnt.loc[len(event_attendee_cnt)] = ['Total', sum(event_attendee_cnt['Attendees'])]
if writer is not None:
    ws.cell(row=5, column=1).value = 'Table FACT_RF_ATTENDEE'
    event_attendee_cnt.to_excel(writer,sheet_name=sheet_name,startrow=5,startcol=0,index=False)
#Not-Registered    14275
#Attended           6888
#Registered         1177
#Cancelled           448


### TABLE ** ATTENDANCE **  ###
event_users = list(attendance.loc[attendance['EVENTCODE'].isin([eventcode]),'ATTENDEE_ID'].unique())    
len(event_users) #7926

#event_users_before with legit sessions
event_users_before = list(set(attendance.loc[(attendance['EVENTCODE'].isin(past_events)) & (attendance['ATTENDEE_ID'].isin(event_users) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID'])))), 'ATTENDEE_ID']))
len(event_users_before)

event_sessions_df = pd.DataFrame(columns=['A','B'])
event_sessions_df.loc[0] = ['event_sessions', len(event_sessions)]
event_sessions_df.loc[1] = ['total users', len(event_users)]
event_sessions_df.loc[2] = ['attended before', len(event_users_before)]
event_sessions_df.loc[3] = ['cold start', len(event_users)-len(event_users_before)]
if writer is not None:
    event_sessions_df.to_excel(writer,sheet_name=sheet_name,startrow=4,startcol=3,index=False)
    ws.cell(row=5, column=4).value = 'Table FACT_RF_SESSION_ATTENDANCE'
    ws.cell(row=5, column=5).value = ''



### SELECT EVENT DATA ###
# Create EVENT_ATTENDANCE #
event_attendance = attendance.loc[(attendance['EVENTCODE'].isin([eventcode])) & (attendance['SESSION_ID'].isin(set(event_sessions['SESSION_ID']))), data_focus_columns['attendance']]
event_attendance['EVENTCODE'].value_counts()
len(set(event_attendance['ATTENDEE_ID']))
event_attendance.shape

event_attendance_df = pd.DataFrame(event_attendance['ATTENDANCE_STATUS'].value_counts()).reset_index()
event_attendance_df.loc[len(event_attendance_df)] = ['Total', sum(event_attendance_df['count'])] 
if writer is not None:
    event_attendance_df.to_excel(writer,sheet_name=sheet_name,startrow=9,startcol=3,index=False)
    ws.cell(row=10, column=4).value = 'ATTENDANCE_STATUS'
    ws.cell(row=10, column=5).value = ''


### Sessions attended/scheduled
event_attendance.value_counts('ATTENDANCE_STATUS')

attendee_session_cnt = pd.DataFrame(event_attendance[event_attendance['ATTENDANCE_STATUS']=='attended'].value_counts('ATTENDEE_ID')).reset_index()
attendee_session_cnt = pd.DataFrame(event_attendance.value_counts('ATTENDEE_ID')).reset_index()
attendee_session_cnt = pd.DataFrame(attendee_session_cnt['count'].value_counts()).rename(columns={'count':'cnt'}).reset_index().rename(columns={'count':'sessions'}).sort_values('sessions')
attendee_session_cnt[:12]
attendee_session_cnt['cnt'].sum()#4210

attendee_session_df = attendee_session_cnt[:12]
attendee_session_df.columns = ['#Sessions attended/scheduled', 'Attendees']
attendee_session_df = attendee_session_df.reset_index(drop=True)

if writer is not None:
    attendee_session_df.to_excel(writer,sheet_name=sheet_name,startrow=16,startcol=3,index=False)    


### Create PAST ATTENDANCE (this is main history for all available users)
past_attendance = attendance.loc[(attendance['EVENTCODE'].isin(past_events)) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))) & (attendance['ATTENDEE_ID'].isin(event_users_before)), data_focus_columns['attendance']]
#len(set(past_attendance['ATTENDEE_ID']))
past_attendance['EVENTCODE'].value_counts()
past_attendance = past_attendance[['EVENTCODE','SESSION_ID','ATTENDEE_ID']]
past_attendance['rating'] = 1.0
#### Mean and Median  number of sessions before excluding cold start
past_attendance_stats = pd.DataFrame([{'metric': 'past_attendance_mean', 'value':round(statistics.mean(past_attendance['ATTENDEE_ID'].value_counts()),1) }])
past_attendance_stats.loc[len(past_attendance_stats)] = {'metric': 'past_attendance_median', 'value':statistics.median(past_attendance['ATTENDEE_ID'].value_counts())}

past_bookmarks = bookmarks.loc[(bookmarks['EVENTCODE'].isin(past_events)) & (bookmarks['SESSION_ID'].isin(set(sessions['SESSION_ID']))) & (bookmarks['ATTENDEE_ID'].isin(event_users_before)), data_focus_columns['bookmarks']]
past_bookmarks['EVENTCODE'].value_counts()
past_bookmarks = past_bookmarks[['EVENTCODE','SESSION_ID','ATTENDEE_ID']]
past_bookmarks['rating'] = 2.0
#### Mean and Median  number of bookmarks before excluding cold start
past_attendance_stats.loc[len(past_attendance_stats)] = {'metric': 'past_bookmarks_mean', 'value':round(statistics.mean(past_bookmarks['ATTENDEE_ID'].value_counts()),1) }
past_attendance_stats.loc[len(past_attendance_stats)] = {'metric': 'past_bookmarks_median', 'value':statistics.median(past_bookmarks['ATTENDEE_ID'].value_counts())}
    
#Join them and delete past_bookmarks
past_attendance = pd.concat([past_attendance,past_bookmarks])
del(past_bookmarks)
past_attendance.shape

if writer is not None:
    past_attendance_stats.to_excel(writer,sheet_name=sheet_name,startrow=33,startcol=3,index=False)


    
#calc_recs = False
###############
# RUN PREDICTION for regular users
ret_recs_size = 30
event_sim_size = 1
t_half = 365

if calc_recs:
    today = pd.Timestamp(event_date).normalize()
    # make sure to take only past sessions
    past_attendance_real = past_attendance.loc[~past_attendance['SESSION_ID'].isin(event_sessions['SESSION_ID'])].copy()
    # cals rating
    past_attendance_real['rating'] = sutil.calc_rating(past_attendance, events, today = today, t_half = t_half)
    event_similarities = session_sims[event_sessions['SESSION_ID']].copy()
    event_similarities = event_similarities.loc[~event_similarities.index.isin(event_sessions['SESSION_ID'])]
    event_similarities = sutil.update_similarities(event_similarities, top_n = event_sim_size, mult = 0.01)
    logger.info("event_similarities size: %s", event_similarities.shape)        
    ret_recs = sutil.calc_user_recs(event_users_before, past_attendance_real, event_similarities, user_sessions_filter, events=None, ret_recs_size=ret_recs_size)
    event_similarities = None
    
    
###### CALC STATS
attendee_hit_ratio, attendee_hit_cnt = sutil.calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
attendee_hit_cnt.columns = ['Number of successful recommendations', 'Attendees']
attendee_hit_ratio.columns = ['Number of recommendations per attendee','PCT of attendees that attended at least at one recommended session']

if writer is not None:
    attendee_hit_cnt[:12].to_excel(writer,sheet_name=sheet_name,startrow=16,startcol=0,index=False)
    ws.cell(row=16, column=1).value = 'HIT Ratio'
    attendee_hit_ratio.to_excel(writer,sheet_name=sheet_name,startrow=32,startcol=0,index=False)
    ws.cell(row=32, column=1).value = 'HIT Ratio'
    
    #writer.close()
    #writer = None


##############################
###### COLD START USERS 
user_id = '1765818109552001ET2S' #rick
# ADD MLI USERS
if add_mli_users:
    event_users = list(set(event_users + attendee_mli['ATTENDEE_ID'].to_list()))
    len(event_users)

cold_start_users = np.setdiff1d(event_users, event_users_before).tolist()
len(cold_start_users) #3840
len(event_attendee.loc[event_attendee['ATTENDEE_ID'].isin(cold_start_users)]) #3405

len(event_attendee)
event_real_attendee = event_attendee[event_attendee['ATTENDEE_ID'].isin(event_users)].reset_index(drop = True)
len(event_real_attendee['ATTENDEE_ID'])
event_real_attendee.iloc[:, :7]
event_real_attendee['description'].iloc[-2]
event_real_attendee.to_csv('~/real_attendee.csv')

#BUILD user_sims
model = SentenceTransformer("all-MiniLM-L6-v2")
user_embeddings = model.encode(event_real_attendee['description'].to_list())
user_embeddings = embeddings(user_embeddings, event_real_attendee['ATTENDEE_ID'])

user_similarities = model.similarity(user_embeddings.embeddings, user_embeddings.embeddings).numpy()
np.fill_diagonal(user_similarities, 0)
#del(user_embeddings)

#BUILD user_similarities
user_similarities = pd.DataFrame(user_similarities, index=user_embeddings.rownames, columns=user_embeddings.rownames)
user_similarities.index.name = user_similarities.columns.name= None
#user_similarities[user_id]

user_similarities = user_similarities.loc[user_similarities.index.isin(cold_start_users)]
user_similarities = user_similarities[np.intersect1d(user_similarities.columns, event_users_before).tolist()]
user_similarities.shape
#user_similarities = user_similarities.stack().reset_index()
#user_similarities.columns = ["ATTENDEE_IN", "ATTENDEE_OUT", "score"]
user_similarities = sutil.matrix2long(user_similarities, top_n=25, batch_size=500, new_cols = ["ATTENDEE_IN", "ATTENDEE_OUT", "score"])
#user_similarities.loc[user_similarities['ATTENDEE_IN']==user_id]


len(set(user_similarities['ATTENDEE_IN']))
len(set(user_similarities['ATTENDEE_OUT']))

###### RUN Cold Start RECS
len(cold_start_users)
#dodac parametr sim_size
ret_recs_size = 30
#calc_recs = True
user_nn_size = 25
sim_size = 1

session_similarities = session_sims[event_sessions['SESSION_ID']].copy()
session_similarities = session_similarities.loc[~session_similarities.index.isin(event_sessions['SESSION_ID'])]
session_similarities = sutil.update_similarities(session_similarities, top_n = sim_size, mult = 0.01)

if calc_recs:
    event_embeddings = session_embeddings.select(list(event_sessions["SESSION_ID"]))
    session_room = session_room.loc[session_room['SESSION_ID'].isin(event_sessions['SESSION_ID'])]
    session_room = session_room.loc[~session_room['SESSION_ID'].isin(['1757341755688001xPnC','1757341479770001hjRw','1757341894165001lPa6'])]
    #ret_recs = sutil.calc_cold_start_recs_nn(cold_start_users, past_attendance, user_similarities, session_similarities, user_nn_size, ret_recs_size)
    ret_recs = sutil.calc_combined_cold_start_recs(user_id, user_similarities, session_similarities, user_embeddings, event_embeddings, past_attendance, session_room, recs_params)
    
len(set(ret_recs['ATTENDEE_ID']))

###### CALC STATS
attendee_hit_ratio, attendee_hit_cnt = sutil.calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
attendee_hit_cnt.columns = ['Number of successful recommendations', 'Cold Start']
attendee_hit_ratio.columns = ['Number of recommendations per attendee','Cold Start']

attendee_hit_cnt
attendee_hit_ratio


if writer is not None:
    attendee_hit_cnt[:12]['Cold Start'].to_excel(writer,sheet_name=sheet_name,startrow=16,startcol=2,index=False)
    #ws.cell(row=16, column=1).value = 'HIT Ratio'    
    attendee_hit_ratio['Cold Start'].to_excel(writer,sheet_name=sheet_name,startrow=32,startcol=2,index=False)
    #ws.cell(row=32, column=1).value = 'HIT Ratio'
    
    writer.close()
    writer = None


#Calc General freq recs
pd.DataFrame(ret_recs['SESSION_ID'].value_counts()).reset_index().rename(columns={'count':'recs'}).merge(
    sessions[['SESSION_ID','TITLE']], how='left', on = 'SESSION_ID').merge(
    pd.DataFrame(attendance.loc[attendance['EVENTCODE']==eventcode,'SESSION_ID'].value_counts()).reset_index(), how='left', on = 'SESSION_ID')[0:50]




##################################
###### CALC RECS for the user
import random
user_id = random.choice(list(set(cold_start_users)))
user_id = '1765818109552001ET2S' #RICK


session_room = session_room.loc[session_room['SESSION_ID'].isin(event_sessions['SESSION_ID'])]
session_room = session_room.loc[~session_room['SESSION_ID'].isin(['1757341755688001xPnC','1757341479770001hjRw','1757341894165001lPa6'])]


sutil.get_nn_users(user_id, user_similarities, event_attendee)
event_embeddings = session_embeddings.select(list(event_sessions["SESSION_ID"]))
event_embeddings.shape()

ret_recs_size = 1600
recs_params["ret_recs_size"] = ret_recs_size

user_ret_recs_tmp = sutil.calc_combined_cold_start_recs(user_id, user_similarities, session_similarities, user_embeddings, event_embeddings, past_attendance, session_room, recs_params)
user_ret_recs_tmp

sutil.show_nice_recs(user_ret_recs_tmp, event_sessions)






user_ret_recs = user_ret_recs_tmp

user_id = '1761600970066001MdwW'
attendee[attendee['EMAIL']=='ben.morris@salesforce.com']
attendee[attendee['EMAIL']=='mgelbman@salesforce.com']

user_id = '1749505213948001egcn'

user_history = sutil.get_user_history(user_id, attendance, bookmarks, sessions, events)
user_history = user_history.loc[user_history['EVENTCODE'].isin(list(past_events) + [eventcode]) ]
user_history = user_history.loc[user_history['ATTENDANCE_STATUS']!='bookmark']
user_history.merge(user_ret_recs[['SESSION_ID','score']], how = 'left', on = ['SESSION_ID'])
user_history



#top sesje dla dreamforce i fykickoff
a = pd.DataFrame(attendance.loc[attendance['EVENTCODE']=='fy27kickoff','SESSION_ID'].value_counts()).reset_index().merge(sessions[['SESSION_ID','TITLE']], on='SESSION_ID', how = 'left')
a[:30]
a = pd.DataFrame(attendance.loc[attendance['EVENTCODE']=='df25','SESSION_ID'].value_counts()).reset_index().merge(sessions[['SESSION_ID','TITLE']], on='SESSION_ID', how = 'left')
a[:30]








user_id = '1765818109552001ET2S' #RICK
user_recs = sutil.calc_cold_start_recs_nn([user_id], past_attendance, user_similarities, event_similarities, user_nn_size, 1600)
user_recs = user_ret_recs_tmp

session_room_user = session_room.merge(user_recs, how = 'left', on = 'SESSION_ID').sort_values(by=["START_TIME"])

session_room_user = session_room_user[['SESSION_ID','START_TIME','END_TIME','DAY_NAME','LENGTH','CAPACITY','score']]
session_room_user[:100]
session_room_user.to_csv('~/session_room_user_2.csv')

schedule = sutil.optimize_session_schedule(session_room_user, start_time='2025-10-14 08:00:00',break_minutes=15,short_session_penalty=0.1)
schedule = schedule.merge(sessions[['SESSION_ID','TITLE']], how='left',on='SESSION_ID')
schedule.to_csv('~/user_schedule_2.csv')
schedule




set(session_room['ROOM'])





#################
#USE OLLAMA TO FIND OUT User desc

rh = attendee[attendee['ATTENDEE_ID'] == '1765818109552001ET2S']
rh[cols] = rh[cols].fillna("")
attendee_desc = sutil.build_attendee_description(rh.iloc[1])
attendee_desc = event_real_attendee['description'].iloc[1]

prompt = "You are a data extractor. Output only factual person information. Never include preambles, disclaimers, or suggestions. Never mention search or lack of search results."
prompt = "Extract only factual information about the person from the provided text. Do not mention search process, uncertainty, apologies, or source availability. Do not say 'I couldn't find'. Do not add commentary, alternatives, or suggestions. Return only a bio. If there is no any information, return exactly: No reliable public information found."

messages = [{
    "role": "user",
    "content": ("Please find professional linkedin information and create short summary about the  " + attendee_desc + ". Please do not write anything about you could't do."),
    },]
response = chat(model="llama3.2:latest", messages=messages)
logger.info("Ollama response: %s", response.message.content)







#################
#len(sessions)
#sessions_times = sutil.parse_times_column(sessions, 'TIMES', 'SESSION_ID')
#sessions_times['SESSION_ID'].value_counts()
#sessions_times.loc[sessions_times['SESSION_ID'].isin(['1693424443460001MHwf'])].to_csv('~/session_times.csv')
#################

####################
meetings = pd.read_csv(data_path + "meetings.csv")
#meetings[:1000].to_csv("~/meetings.csv")
meetings = meetings.rename(columns = {'PARTICIPANT_ATTENDEEID':'ATTENDEE_ID'})
meetings = meetings[data_focus_columns['meetings']]
meetings['START_TIME_UTC'] = pd.to_datetime(meetings['START_TIME_UTC'], utc=True)
meetings = meetings[meetings['START_TIME_UTC'] >= '2024-05-01']

#meetings = meetings[:10000]
len(set(meetings['SESSION_ID']))
len(set(meetings['ATTENDEE_ID']))


###################
sim_users = sutil.calc_user_similarities_from_attendance(meetings, top_n=10, batch_size=1000)
sim_users = sutil.calc_user_similarities_from_attendance(past_attendance, top_n=10, batch_size=1000)
