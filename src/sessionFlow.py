#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:45:39 2026

@author: mdraminski
"""

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
from sfconfig import sfconfig
from sfloader import sfloader
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

config = sfconfig()
data_path = config.data_path
tmp_path = config.tmp_path
results_path = config.results_path
attendee_mli_email = config.attendee_mli_email
data_focus_columns = config.data_focus_columns
recs_params = config.get_params()
loader = sfloader(config)

#Ben Morris
#David Keane
#Matt Gelbman
#Rick Handt
#Surabhi Shastri

############################################
### LOAD AND PREPARE THE DATA ###
############################################
events = loader.load_events()
events_start_datetime = loader.create_events_start_datetime(events)
sessions = loader.load_sessions()
speakers = loader.load_speakers()
session_room = loader.load_session_room()
session_start_datetime = loader.create_session_start_datetime(session_room)
attendee = loader.load_attendee()
attendee_mli = loader.create_attendee_mli(attendee)
attendance = loader.load_attendance(sessions, session_start_datetime, events_start_datetime)
bookmarks = loader.load_bookmarks(attendee, sessions, attendance, events_start_datetime)
attendance = loader.append_bookmarks_to_attendance(attendance, bookmarks)
bookmarks = None


############################################
### LOAD EVENTS
events = pd.read_csv(data_path + "events.csv")
events = events[data_focus_columns['events']]
events.shape

### CREATE EVENTS_START_DATETIME
events_start_datetime = events[['EVENTCODE','START_DATE']]
events_start_datetime = events_start_datetime.rename(columns={'START_DATE':'EVENT_DATETIME'})
events_start_datetime["EVENT_DATETIME"] = (pd.to_datetime(events_start_datetime["EVENT_DATETIME"], errors="coerce").dt.normalize()+ pd.Timedelta(hours=9))
events_start_datetime

### LOAD SESSIONS
sessions = pd.read_csv(data_path + "sessions.csv")
sessions = sessions[data_focus_columns['sessions']]
sessions = sutil.session_filter(sessions, True, version = 2)
sessions.iloc[1]
sessions
logger.info(util.show_object_memory(sessions, "sessions"))

### LOAD SPEAKERS
speakers = pd.read_csv(data_path + "speakers.csv")
speakers = speakers[data_focus_columns['speakers']]
speakers
logger.info(util.show_object_memory(speakers, "speakers"))

### LOAD SESSION_ROOM
session_room = pd.read_csv(data_path + "session_room.csv")
session_room = session_room[data_focus_columns['session_room']]
session_room["START_TIME"] = (pd.to_datetime(session_room["UTC_START_TIME"], utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
session_room["END_TIME"] = (pd.to_datetime(session_room["UTC_END_TIME"], utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
session_room = session_room.drop(columns=["UTC_START_TIME", "UTC_END_TIME"])
session_room
session_room.iloc[1]
logger.info(util.show_object_memory(session_room, "session_room"))

### CREATE SESSION_STARTTIME
session_start_datetime = session_room.groupby(["EVENTCODE", "SESSION_ID"], as_index=False)["START_TIME"].min()
session_start_datetime = session_start_datetime.rename(columns={'START_TIME':'SESSION_DATETIME'})

### LOAD ATTENDEE
attendee = pd.read_csv(data_path + "attendee.csv")
attendee = attendee.loc[attendee['TEST_RECORD'] != 'Yes']
attendee = attendee[data_focus_columns['attendee']]
attendee
logger.info(util.show_object_memory(attendee, "attendee"))

attendee_mli = attendee.loc[attendee['EMAIL'].isin(attendee_mli_email), ['ATTENDEE_ID','EMAIL']]
attendee_mli = attendee_mli.drop_duplicates()

### LOAD ATTENDANCE
attendance = pd.read_csv(data_path + "attendance.csv")
#attendance[:3000].to_csv('~/attendance3000.csv')
attendance = attendance[data_focus_columns['attendance']]
#only legit sessions
attendance = attendance.loc[attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))]
attendance["ATTENDED_DATETIME"] = pd.to_datetime(attendance["ATTENDEDDATE"].astype("string") + " " + attendance["ATTENDEDTIME"].astype("string"), errors="coerce")
attendance = attendance[['EVENTCODE','SESSION_ID','ATTENDEE_ID','ATTENDED_DATETIME','ATTENDANCE_STATUS']]
attendance = attendance.merge(session_start_datetime, how = 'left', on = ['EVENTCODE','SESSION_ID'])
attendance = attendance.merge(events_start_datetime, how = 'left', on = ['EVENTCODE'])
attendance.loc[attendance['SESSION_DATETIME'].isnull(),'SESSION_DATETIME'] = attendance.loc[attendance['SESSION_DATETIME'].isnull(),'EVENT_DATETIME']
attendance.loc[~attendance['ATTENDED_DATETIME'].isnull(), 'ATTENDANCE_STATUS'] = 'attended'
attendance.loc[attendance['ATTENDANCE_STATUS'].isnull(),'ATTENDANCE_STATUS'] = 'scheduled'
attendance = attendance[['EVENTCODE','SESSION_ID','ATTENDEE_ID','ATTENDED_DATETIME','SESSION_DATETIME','ATTENDANCE_STATUS']]
logger.info(attendance['ATTENDANCE_STATUS'].value_counts())
logger.info(util.show_object_memory(attendance, "attendance"))


### LOAD BOOKMARKS
bookmarks = pd.read_csv(data_path + "bookmarks.csv")
# add ATTENDEE_ID to bookmarks coz these is not
bookmarks = bookmarks.merge(attendee[['ATTENDEE_ID','EMAIL']].drop_duplicates(), how = 'left', on = 'EMAIL')
bookmarks = bookmarks[~bookmarks['ATTENDEE_ID'].isnull()]
bookmarks = bookmarks[data_focus_columns['bookmarks']]
#bookmarks[:1000].to_csv("~/bookmarks.csv")
# SOME SESSIONS MAY occur many times on one conference
bookmarks.loc[bookmarks['SESSION_ID']=='1722538248635001Pgq8',['EVENTCODE','SESSION_ID','SESSION_TIME','SESSION_DATE', 'SESSION_TITLE']].drop_duplicates().sort_values(by=['SESSION_DATE','SESSION_TIME'])
bookmarks = bookmarks[bookmarks['SESSION_ID'].isin(sessions['SESSION_ID'])]
logger.info(util.show_object_memory(bookmarks, "bookmarks"))

bookmarks = bookmarks.rename(columns={"SCHEDULED_DATETIME":"SESSION_DATETIME"})
bookmarks["SESSION_DATETIME"] = pd.to_datetime(bookmarks["SESSION_DATETIME"],errors="coerce")
bookmarks['ATTENDANCE_STATUS'] = 'bookmark'
bookmarks = bookmarks[attendance.columns]
bookmarks = bookmarks.merge(events_start_datetime, how = 'left', on = ['EVENTCODE'])
bookmarks.loc[bookmarks['SESSION_DATETIME'].isnull(),'SESSION_DATETIME'] = bookmarks.loc[bookmarks['SESSION_DATETIME'].isnull(),'EVENT_DATETIME']
bookmarks = bookmarks[attendance.columns]
attendance = pd.concat([attendance,bookmarks]).sort_values(by=['EVENTCODE','SESSION_DATETIME'])
logger.info(util.show_object_memory(attendance, "attendance"))


############################################
### Calculate SESSIONS EMBEDDINGS
############################################
sentenceTransformerModel = SentenceTransformer("all-MiniLM-L6-v2")

session_embeddings = sutil.calc_session_embeddings(sessions, speakers)
logger.info("Embeddings shape: %s", session_embeddings.shape)
type(session_embeddings)
session_embeddings = embeddings(session_embeddings, sessions['SESSION_ID'])
session_embeddings.shape()
logger.info(util.show_object_memory(session_embeddings, "session_embeddings"))

### Calculate session_similarity based on Embedding
session_similarity = sentenceTransformerModel.similarity(session_embeddings.embeddings, session_embeddings.embeddings).numpy()
logger.info("Similarity matrix shape: %s", session_similarity.shape)
session_similarity = pd.DataFrame(session_similarity, index=sessions['SESSION_ID'], columns=sessions['SESSION_ID'])
session_similarity.index.name = session_similarity.columns.name= None
logger.info("Similarity session_sims matrix:\n%s", session_similarity.shape)
logger.info(util.show_object_memory(session_similarity, "session_similarity"))

############################################
# THE EXPERIMENT
############################################
eventcode = 'ghost_event'
add_mli_users = True
#eventcode = 'tdx26'

event = events.loc[events['EVENTCODE']==eventcode, data_focus_columns['events']]
_event_end_dates = events.loc[events['EVENTCODE'].isin([eventcode]),'END_DATE']
if _event_end_dates.empty:
    raise ValueError(f"No event found for eventcode={eventcode!r}")
event_date = _event_end_dates.iloc[0]
past_events = events.loc[events['END_DATE'] < event_date,'EVENTCODE']
#events.loc[events['EVENTCODE'].isin(past_events)].to_csv('~/events.csv')

### SELECT EVENT SESSIONS DATA
event_sessions = sessions.loc[sessions['EVENTCODE'].isin([eventcode])].reset_index()
event_sessions['SESSION_TYPE'].value_counts()
logger.info(util.show_object_memory(event_sessions, "event_sessions"))

### SELECT EVENT ATTENDEE DATA
event_attendee = attendee.loc[attendee['EVENTCODE']==eventcode, data_focus_columns['attendee']]
#ADD MLI USERS
if add_mli_users:
    event_attendee_mli = attendee.loc[attendee['ATTENDEE_ID'].isin(attendee_mli['ATTENDEE_ID']),event_attendee.columns].groupby(["ATTENDEE_ID"]).last().reset_index()
    event_attendee_mli['EVENTCODE'] = eventcode
    event_attendee_mli['STATUS'] = 'Attended'    
    event_attendee = pd.concat([event_attendee, event_attendee_mli])
    del(event_attendee_mli)
#event_attendee[:1000].to_csv('~/attendee.csv')
event_attendee
#logger.info(event_attendee['STATUS'].value_counts())
logger.info(util.show_object_memory(event_attendee, "event_attendee"))

### CALC ATTENDEE_EMBEDDINGS
#ADD attendee description
attendee_description_columns = ["FIRST_NAME", "LAST_NAME", "COMPANY_NAME","JOB_TITLE", "PRIMARY_ROLE", "PRIMARY_INDUSTRY"]
event_attendee[attendee_description_columns] = event_attendee[attendee_description_columns].fillna("")
event_attendee["description"] = event_attendee.apply(sutil.build_attendee_description, axis=1)
event_attendee["description"] = (event_attendee["description"].fillna("").astype(str).str.strip())
event_attendee['description'].iloc[1]
logger.info(event_attendee['STATUS'].value_counts())
logger.info(util.show_object_memory(event_attendee, "event_attendee"))
#event_attendee.to_csv('~/real_attendee.csv')
#BUILD attendee_embeddings
attendee_embeddings = sentenceTransformerModel.encode(event_attendee['description'].to_list())
attendee_embeddings = embeddings(attendee_embeddings, event_attendee['ATTENDEE_ID'])
logger.info(util.show_object_memory(attendee_embeddings, "attendee_embeddings"))
attendee_embeddings.shape()
attendee_embeddings

### GET RETURNING AND COLD START USERS FROM attendance
event_users_id = list(attendance.loc[attendance['EVENTCODE'].isin([eventcode]),'ATTENDEE_ID'].unique())
if add_mli_users:
    event_users_id = event_users_id + attendee_mli['ATTENDEE_ID'].to_list()
event_attendee_id = list(pd.Series(event_users_id)[pd.Series(event_users_id).isin(event_attendee['ATTENDEE_ID'])])
#event_users = event_users_id
event_users_returning_id = list(set(attendance.loc[(attendance['EVENTCODE'].isin(past_events)) & (attendance['ATTENDEE_ID'].isin(event_users_id) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID'])))), 'ATTENDEE_ID']))
#event_users_before = event_users_returning_id
event_users_coldstart_id = np.setdiff1d(event_users_id, event_users_returning_id).tolist()
logger.info("#event_users: %s #returning users: %s #coldstart users: %s #event attendees: %s", len(event_users_id),len(event_users_returning_id),len(event_users_coldstart_id),len(event_attendee_id))
#cold_start_users = event_users_coldstart_id

### GET past_attendance AND event_attendance
event_attendance = attendance.loc[(attendance['EVENTCODE'].isin([eventcode])) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))),]
logger.info(util.show_object_memory(event_attendance, "event_attendance"))                                   
past_attendance = attendance.loc[(attendance['EVENTCODE'].isin(past_events)) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))) & (attendance['ATTENDEE_ID'].isin(event_users_returning_id)),]
logger.info(util.show_object_memory(past_attendance, "past_attendance"))

#### Mean and Median number of sessions before excluding cold start
past_attendance_stats = pd.DataFrame([{'metric': 'past_attendance_mean', 'value':round(statistics.mean(past_attendance.loc[past_attendance["ATTENDANCE_STATUS"].isin(['scheduled','attended','waitlisted']),'ATTENDEE_ID'].value_counts()),1) },
                                      {'metric': 'past_attendance_median', 'value':round(statistics.median(past_attendance.loc[past_attendance["ATTENDANCE_STATUS"].isin(['scheduled','attended','waitlisted']),'ATTENDEE_ID'].value_counts()),1) },
                                      {'metric': 'past_bookmarks_mean', 'value':round(statistics.mean(past_attendance.loc[past_attendance["ATTENDANCE_STATUS"].isin(['bookmark']),'ATTENDEE_ID'].value_counts()),1) },
                                      {'metric': 'past_bookmarks_median', 'value':round(statistics.median(past_attendance.loc[past_attendance["ATTENDANCE_STATUS"].isin(['bookmark']),'ATTENDEE_ID'].value_counts()),1) }])

ret_recs_size = recs_params['ret_recs_size']
today = pd.Timestamp(event_date).normalize()
# cals rating
past_attendance['rating'] = sutil.calc_rating(past_attendance, events, today=today, recs_params=recs_params)
event_attendance['rating'] = sutil.calc_rating(event_attendance, events, today=today, recs_params=recs_params)

#select event_session_similarity
event_session_similarity = session_similarity[event_sessions['SESSION_ID']].copy()
event_session_similarity = event_session_similarity.loc[~event_session_similarity.index.isin(event_sessions['SESSION_ID'])]
event_session_similarity = sutil.update_similarities(event_session_similarity, recs_params)
logger.info("event_session_similarity size: %s", event_session_similarity.shape)

### GET Session Filter ###
user_sessions_filter = sutil.get_sessions_filter(event_attendee, event_sessions)

###### CALC RET_RECS - candidates recs based on past_attendance
ret_recs = sutil.calc_users_recs(
    event_users_returning_id,
    past_attendance,
    event_session_similarity,
    user_sessions_filter=user_sessions_filter,
    events=None,
    recs_params=recs_params,
)
#event_session_similarity = None
#calc event_user_embeddings based on past_attendance
event_user_embeddings = sutil.calc_users_embeddings(event_users_returning_id, past_attendance, session_embeddings)


###### CALC RET_RECS STATS
attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = sutil.calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
attendee_hit_cnt.columns = ['Number of successful recommendations', 'Attendees']
attendee_hit_ratio.columns = ['Number of recommendations per attendee','PCT of attendees that attended at least at one recommended session']
ndcg = sutil.calc_ndcg(ret_recs, event_attendance, recs_top_n=15, return_details = True)
ndcg[0]


###################################
### CALC USER_SIMILARITY AND EVENT_USER_SIMILARITY # both based on attendance
user_similarity = sutil.calc_user_similarity_from_attendance(past_attendance, top_n=20, batch_size=1000)
event_user_similarity = sutil.calc_user_similarity_from_attendance(event_attendance.loc[event_attendance['ATTENDEE_ID'].isin(past_attendance['ATTENDEE_ID'])], top_n=20, batch_size=1000)
common_users = list(set(user_similarity["ATTENDEE_IN"]).intersection(set(event_user_similarity["ATTENDEE_IN"])))
print(len(set(user_similarity["ATTENDEE_IN"])), len(set(event_user_similarity["ATTENDEE_IN"])), len(common_users))
summary, details, overlap_pairs = sutil.compare_similar_users(user_similarity, event_user_similarity, top_n=20, return_details=True)
summary
details

### CALC ATTENDEE_SIMILARITY # based on attendance
len(event_users_returning_id)
len(event_users_coldstart_id)
attendee_similarity = sutil.calc_attendee_similarity(event_users_returning_id, event_users_coldstart_id, attendee_embeddings, sentenceTransformerModel, cold_start_batch_size=1000, top_n=recs_params['user_sim_size'], matrix_batch_size=500, progress_every=recs_params['progress_every'])
len(set(attendee_similarity['ATTENDEE_IN']))
len(set(attendee_similarity['ATTENDEE_OUT']))


###################################
###### RUN Cold Start RECS
ret_recs_size = recs_params['ret_recs_size']

#select event_session_similarity
event_session_similarity = session_similarity[event_sessions['SESSION_ID']].copy()
event_session_similarity = event_session_similarity.loc[~event_session_similarity.index.isin(event_sessions['SESSION_ID'])]
event_session_similarity = sutil.update_similarities(event_session_similarity, recs_params)
logger.info("event_session_similarity size: %s", event_session_similarity.shape)


session_room = session_room.loc[session_room['SESSION_ID'].isin(event_sessions['SESSION_ID'])]
session_room = session_room.loc[~session_room['SESSION_ID'].isin(['1757341755688001xPnC','1757341479770001hjRw','1757341894165001lPa6'])]
ret_recs_popular = sutil.get_popular_sessions(event_sessions, session_room, user_sessions_filter, recs_params, ret_recs_size).copy()


event_session_embeddings = session_embeddings.select(list(event_sessions["SESSION_ID"]))
user_ret_recs_nn = sutil.calc_cold_start_recs_nn(event_users_returning_id, past_attendance, attendee_similarity, event_session_similarity, user_sessions_filter, recs_params, ret_recs_size).copy()
user_ret_recs_emb = sutil.calc_cold_start_recs_emb(event_users_returning_id, attendee_embeddings, event_session_embeddings, user_sessions_filter, recs_params, ret_recs_size).copy()


attendee_mli.loc[attendee_mli['EMAIL']=='user1@example.com','ATTENDEE_ID']

user_id = '1765818109552001ET2S' #   user1@example.com
#user_id = '1772647553672001tEe7' #   user4@example.com

attendee_similarity[attendee_similarity['ATTENDEE_IN'].isin([user_id])].merge(event_attendee[['ATTENDEE_ID','description']], how='left', left_on='ATTENDEE_OUT', right_on='ATTENDEE_ID')


#event_sessions.merge(session_room, on='SESSION_ID', how='left').to_csv("~/sessiosns.csv")
#event_sessions['SESSION_TYPE'].value_counts()

ret_recs_size = recs_params['ret_recs_size']
ret_recs_popular = sutil.get_popular_sessions(event_sessions, session_room, user_sessions_filter, recs_params, ret_recs_size=ret_recs_size * recs_params['ret_recs_size_mult']).copy()
sutil.show_nice_recs(ret_recs_popular, event_sessions)
user_ret_recs_popular = sutil.get_recs_popular_for_users([user_id], ret_recs_popular, user_sessions_filter, ret_recs_size)
sutil.show_nice_recs(user_ret_recs_popular, event_sessions)

user_ret_recs_nn = sutil.calc_cold_start_recs_nn([user_id], past_attendance, attendee_similarity, event_session_similarity, user_sessions_filter, recs_params, ret_recs_size).copy()
sutil.show_nice_recs(user_ret_recs_nn, event_sessions)

user_ret_recs_emb = sutil.calc_cold_start_recs_emb([user_id], attendee_embeddings, event_session_embeddings, user_sessions_filter, recs_params, ret_recs_size).copy()
sutil.show_nice_recs(user_ret_recs_emb, event_sessions)


user_id = '1765818109552001ET2S' #   user1@example.com
user_recs = sutil.calc_combined_cold_start_recs(user_id, attendee_similarity, event_session_similarity, attendee_embeddings, event_session_embeddings, past_attendance, event_sessions, session_room, user_sessions_filter, recs_params)
sutil.show_nice_recs(user_recs[0:ret_recs_size], event_sessions)

user_id = '1772647553672001tEe7' #   user4@example.com
user_recs = sutil.calc_combined_cold_start_recs(user_id, attendee_similarity, event_session_similarity, attendee_embeddings, event_session_embeddings, past_attendance, event_sessions, session_room, user_sessions_filter, recs_params)
sutil.show_nice_recs(user_recs[0:ret_recs_size], event_sessions)





session_room_user = session_room.merge(user_recs, how = 'left', on = 'SESSION_ID').sort_values(by=["START_TIME"])
session_room_user = session_room_user[['SESSION_ID','START_TIME','END_TIME','DAY_NAME','LENGTH','CAPACITY','score']]
session_room_user.to_csv(results_path + 'session_room_user_2.csv')

schedule = sutil.optimize_session_schedule(session_room_user, start_time='2025-10-14 08:00:00',break_minutes=15,short_session_penalty=0.1)
schedule = schedule.merge(sessions[['SESSION_ID','TITLE']], how='left',on='SESSION_ID')
schedule.to_csv(results_path + 'user_schedule_2.csv')
schedule





#session_embeddings.shape()
#session_similarity
#event_session_embeddings
#event_session_similarity #tmp

#attendee_embeddings.shape() # based on attendee description
#attendee_similarity # based on attendee description

#event_user_embeddings # based on attendance
#user_similarity # based on attendance


import importlib
import sfutils as sutil
import utils as util
import sfloader

importlib.reload(sutil)
importlib.reload(sfloader)
loader = sfloader.sfloader(config)
