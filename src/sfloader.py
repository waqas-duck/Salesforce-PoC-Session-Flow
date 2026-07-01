#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV loading and preprocessing for SessionFlow.
"""

import logging
import os
import utils as util
import pandas as pd
from sfconfig import sfconfig

logger = logging.getLogger(__name__)


class sfloader:
    def __init__(self, config=None):
        self.config = config if config is not None else sfconfig()        
        self.data_path = self.config.data_path
        self.data_focus_columns = self.config.data_focus_columns
        self.attendee_mli_email = self.config.attendee_mli_email

    def load_csv(self, filename, **read_csv_kwargs):
        return pd.read_csv(os.path.join(self.data_path, filename), **read_csv_kwargs)

    ###############################
    ### LOAD EVENTS    
    def load_events(self):
        return self.process_events(self.load_csv('events.csv'))

    def process_events(self, events):
        events = events[self.data_focus_columns['events']].copy()
        logger.info(util.show_object_memory(events, "events"))
        return events
    
    ###############################
    ### CREATE EVENTS_START_DATETIME
    def create_events_start_datetime(self, events):
        events_start_datetime = events[['EVENTCODE','START_DATE']].copy()
        events_start_datetime = events_start_datetime.rename(columns={'START_DATE':'EVENT_DATETIME'})
        events_start_datetime["EVENT_DATETIME"] = (pd.to_datetime(events_start_datetime["EVENT_DATETIME"], errors="coerce").dt.normalize()+ pd.Timedelta(hours=9))                
        return events_start_datetime

    ###############################
    ### LOAD SESSIONS
    def load_sessions(self):
        sessions = self.process_sessions(
            self.load_csv(
                'sessions.csv',
                dtype={
                    'MISSED_SESSION': 'string',
                    'SESSION_SUMMARY_BY_EINSTEIN': 'string',
                    'EINSTEIN_SUMMARIES_AGENCY_TRACKING': 'string',
                    'KEY_TAKEAWAYS_BY_EINSTEIN': 'string',
                    'CHANGES_MADE': 'string',
                    'CONTENT_SESSION_LEVEL': 'string',
                },
            )
        )
        return sessions

    def process_sessions(self, sessions):
        sessions = sessions[self.data_focus_columns['sessions']].copy()
        sessions = self._session_filter(sessions, True, version = 2)        
        logger.info(util.show_object_memory(sessions, "sessions"))
        return sessions

    def _session_filter(self, df, keep=True, version = 2):
        if version == 1:
            mask = (
                df['SESSIONCODE'].astype(str).str.startswith(('STAFF-', 'KeySec-', 'DRY_RUN'), na=False)
                | df['TITLE'].astype(str).str.startswith(('STAFF-', 'KeySec-', 'Dry Run Test', 'Lunch'), na=False)
                | df['ABSTRACT'].astype(str).str.startswith(('Quest', 'Test'), na=False)
                | df['TYPE'].astype(str).str.startswith(('Meal', 'Breakout', 'Security'), na=False)
                | df['STATUS'].astype(str).str.startswith(('Cancelled', 'Rejected'), na=False)
                | (df['PUBLISHED'] == False)
            )
            # Keep rows that DO NOT match the conditions
            ret_df = df[~mask] if keep else df[mask]
        elif version == 2:
            session_types = [
                "Main Keynote",
                "Keynote",
                "Activity",
                "Salesforce+ Stage",
                "Theater",
                "Breakout",            
                "Hands-on Training",
                "Roundtable",
                "Workshop",
                "Certification",
                "Community Networking",
                "Executive Summit",]        
            ret_df = df[df['SESSION_TYPE'].isin(session_types)]
        else:
            logger.warning("version parameter is invalid: %s", version)
            ret_df = None
            
        ret_df = ret_df.loc[(ret_df['STATUS'].isin(['Accepted'])) & (ret_df['PUBLISHED'] == True)]
        ret_df = ret_df.loc[~ret_df['SESSIONCODE'].str.startswith('STAFF')]    
        return ret_df
                
    ###############################
    ### LOAD SPEAKERS    
    def load_speakers(self):
        speakers = self.process_speakers(self.load_csv('speakers.csv'))        
        return speakers
            
    def process_speakers(self, speakers):        
        speakers = speakers[self.data_focus_columns['speakers']].copy()        
        logger.info(util.show_object_memory(speakers, "speakers"))
        return speakers

    ###############################
    ### LOAD SESSION_ROOM
    def load_session_room(self):
        session_room = self.process_session_room(self.load_csv('session_room.csv'))        
        return session_room

    def process_session_room(self, session_room):        
        session_room = session_room[self.data_focus_columns['session_room']].copy()
        session_room["START_TIME"] = (pd.to_datetime(session_room["UTC_START_TIME"], utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
        session_room["END_TIME"] = (pd.to_datetime(session_room["UTC_END_TIME"], utc=True).dt.tz_convert("America/Los_Angeles").dt.tz_localize(None))
        session_room = session_room.drop(columns=["UTC_START_TIME", "UTC_END_TIME"])
        logger.info(util.show_object_memory(session_room, "session_room"))
        return session_room

    ###############################
    ### CREATE session_start_datetime
    def create_session_start_datetime(self, session_room):
        session_start_datetime = session_room.groupby(["EVENTCODE", "SESSION_ID"], as_index=False)["START_TIME"].min()
        session_start_datetime = session_start_datetime.rename(columns={'START_TIME':'SESSION_DATETIME'})
        return session_start_datetime

    ###############################
    ### LOAD ATTENDEE
    def load_attendee(self):
        attendee = self.process_attendee(self.load_csv('attendee.csv'))        
        return attendee

    def process_attendee(self, attendee):        
        attendee = attendee.loc[attendee['TEST_RECORD'] != 'Yes'].copy()
        attendee = attendee[self.data_focus_columns['attendee']]
        logger.info(util.show_object_memory(attendee, "attendee"))
        return attendee

    ###############################
    ### CREATE attendee_mli
    def create_attendee_mli(self, attendee):
        attendee_mli = attendee.loc[attendee['EMAIL'].isin(self.attendee_mli_email), ['ATTENDEE_ID','EMAIL']]
        attendee_mli = attendee_mli.drop_duplicates()
        return attendee_mli

    ###############################
    ### LOAD ATTENDANCE
    def load_attendance(self, sessions, session_start_datetime, events_start_datetime):
        attendance = self.process_attendance(self.load_csv('attendance.csv'),sessions,session_start_datetime,events_start_datetime)
        return attendance

    def process_attendance(self, attendance, sessions, session_start_datetime, events_start_datetime):                        
        attendance = attendance[self.data_focus_columns['attendance']].copy()
        #only legit sessions
        attendance = attendance.loc[attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))].copy()
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
        
        return attendance
        
    ###############################
    ### LOAD BOOKMARKS
    def load_bookmarks(self, attendee, sessions, attendance, events_start_datetime):
        return self.process_bookmarks(self.load_csv('bookmarks.csv'),attendee,sessions,attendance,events_start_datetime)

    def process_bookmarks(self, bookmarks, attendee, sessions, attendance, events_start_datetime):
        # add ATTENDEE_ID to bookmarks coz these is not
        bookmarks = bookmarks.merge(attendee[['ATTENDEE_ID','EMAIL']].drop_duplicates(), how = 'left', on = 'EMAIL')
        bookmarks = bookmarks[~bookmarks['ATTENDEE_ID'].isnull()].copy()
        bookmarks = bookmarks[self.data_focus_columns['bookmarks']].copy()
        #bookmarks[:1000].to_csv("~/bookmarks.csv")
        
        # SOME SESSIONS MAY occur many times on one conference
        bookmarks.loc[bookmarks['SESSION_ID']=='1722538248635001Pgq8',['EVENTCODE','SESSION_ID','SESSION_TIME','SESSION_DATE', 'SESSION_TITLE']].drop_duplicates().sort_values(by=['SESSION_DATE','SESSION_TIME'])
        bookmarks = bookmarks[bookmarks['SESSION_ID'].isin(sessions['SESSION_ID'])].copy()
        logger.info(util.show_object_memory(bookmarks, "bookmarks"))
        
        bookmarks = bookmarks.rename(columns={"SCHEDULED_DATETIME":"SESSION_DATETIME"})
        bookmarks["SESSION_DATETIME"] = pd.to_datetime(bookmarks["SESSION_DATETIME"],errors="coerce")
        bookmarks['ATTENDANCE_STATUS'] = 'bookmark'
        bookmarks = bookmarks[attendance.columns]
        bookmarks = bookmarks.merge(events_start_datetime, how = 'left', on = ['EVENTCODE'])
        bookmarks.loc[bookmarks['SESSION_DATETIME'].isnull(),'SESSION_DATETIME'] = bookmarks.loc[bookmarks['SESSION_DATETIME'].isnull(),'EVENT_DATETIME']
        bookmarks = bookmarks[attendance.columns]
        logger.info(util.show_object_memory(bookmarks, "bookmarks"))
        
        return bookmarks

    ###############################
    ### APPEND bookmarks to attendance
    def append_bookmarks_to_attendance(self, attendance, bookmarks):
        attendance = pd.concat([attendance,bookmarks]).sort_values(by=['EVENTCODE','SESSION_DATETIME'])
        logger.info(util.show_object_memory(attendance, "attendance"))
        return attendance

    ###############################
    ### DOWNLOAD DATA FROM SNOWFLAKE
    def download_data(self, datasets=None):
        import snowflake.connector as sf

        conn_params = {
            'account': 'SFDC_DP_PRD',
            'user': 'SVC_BT_BU_MLI_DEVELOPER_PRD',
            'authenticator': 'SNOWFLAKE_JWT',
            'private_key': util.get_private_key_from_secrets_manager(),
            'warehouse': 'WH_MLI_ETL',
            'database': 'SSE_DM_MKT_PRD',
            'schema': 'WRK_TRUTH_PROFILE'
        }
        ctx = sf.connect(**conn_params)
        cs = ctx.cursor()

        data_query = {
            "events":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_EVENT;",
            "sessions":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION;",
            "speakers":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION_SPEAKER WHERE SPEAKER_ID is not NULL;",
            "attendee":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_ATTENDEE WHERE ATTENDEE_ID is not NULL;",
            "session_room":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_ROOM_TIME;",
            "meetings":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_MEETING;",
            "attendance":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_ATTENDANCE;",
            "bookmarks":"select * from SSE_DM_MKT_PRD.DM_MARKETING.DIM_RF_SESSION_BOOKMARKS;",
            "favorites":"select * from SSE_DM_MKT_PRD.DM_MARKETING.FACT_RF_SESSION_FAVORITES;",
        }

        drop_columns_all = ['FILES', 'FILE_NAME','FILE_ROW_NUMBER','FILE_LAST_MODIFIED','AUDIT_ETL_JOB_INS_TS','AUDIT_ETL_JOB_UPD_TS']
        data_drop_columns = {
            "events":drop_columns_all,
            "sessions":drop_columns_all,
            "speakers":drop_columns_all,
            "attendee":drop_columns_all,
            "session_room":drop_columns_all,
            "meetings":drop_columns_all,
            "attendance":drop_columns_all,
            "bookmarks":drop_columns_all,
            "favorites":drop_columns_all,
        }

        if datasets is None:
            datasets = data_query.keys()

        try:
            for dt in datasets:
                logger.info("Fetching dataset: %s", dt)
                df = util.fetch_pandas_all(cs, data_query[dt])
                df = df.drop(columns=data_drop_columns[dt], errors='ignore')
                if dt == 'sessions':
                    self._session_filter(df, True).to_csv(os.path.join(self.data_path, dt + "_filtered.csv"), index=False)
                df.to_csv(os.path.join(self.data_path, dt + ".csv"), index=False)
        finally:
            cs.close()
            ctx.close()
