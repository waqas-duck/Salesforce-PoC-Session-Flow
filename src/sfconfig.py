#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration defaults for SessionFlow.
"""

from copy import deepcopy


class sfconfig:
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.data_path = '/Users/mdraminski/Workspace/SessionFlow/data/'
        self.tmp_path = '/Users/mdraminski/TEMP4/'
        self.results_path = '/Users/mdraminski/Workspace/SessionFlow/Results/'
        self.attendee_mli_email = [
            'user1@example.com',
            'user2@example.com',
            'user3@example.com',
            'user4@example.com',
        ]

        self.data_focus_columns = {
            'events': [
                'EVENTCODE', 'NAME', 'DISPLAY_NAME', 'CITY', 'TYPE', 'START_DATE', 'END_DATE',
                'COUNTRY_ID', 'STATE_ID', 'STATUS', 'VENUE_NAME', 'DESCRIPTION',
            ],
            'sessions': [
                'SESSION_ID', 'SESSIONCODE', 'TITLE', 'ABSTRACT', 'LENGTH', 'EVENTCODE',
                'EVENT_NAME', 'EVENT_ID', 'TIMES_OFFERED', 'SESSION_TYPE', 'STATUS',
                'PUBLISHED', 'SESSION_TRACK', 'SESSION_SUMMARY_BY_EINSTEIN',
                'KEY_TAKEAWAYS_BY_EINSTEIN', 'SESSION_MEDIUM',
            ],
            'speakers': [
                'EVENTCODE', 'SESSION_ID', 'SPEAKER_ID', 'ATTENDEE_ID', 'GLOBAL_FULL_NAME',
                'GLOBAL_COMPANY', 'GLOBAL_JOB_TITLE', 'BIO',
            ],
            'attendee': [
                'EVENTCODE', 'ATTENDEE_ID', 'STATUS', 'JOB_TITLE', 'FIRST_NAME', 'LAST_NAME',
                'EMAIL', 'COMPANY_NAME', 'ATTENDEE_TYPE', 'CANCELLED', 'TEST_RECORD',
                'APPROVAL_STATUS', 'REGISTERED', 'CHECKIN_DATE', 'PRIMARY_ROLE',
                'PRIMARY_INDUSTRY',
            ],
            'session_room': [
                'EVENTCODE', 'SESSION_ID', 'ROOM_ID', 'ROOM', 'START_TIME', 'END_TIME',
                'DAY_NAME', 'LENGTH', 'CAPACITY', 'UTC_START_TIME', 'UTC_END_TIME',
            ],
            'meetings': [
                'EVENTCODE', 'ATTENDEE_ID', 'SESSION_ID', 'CODE', 'PARTICIPANT_COMPANY',
                'PARTICIPANT_EMAIL', 'PARTICIPANT_FIRST_NAME', 'PARTICIPANT_LAST_NAME',
                'PARTICIPANT_ROLES', 'TITLE', 'ABSTRACT', 'STATUS', 'MEETING_PROGRAM_ID',
                'MEETING_PROGRAM_NAME', 'ROOM', 'MEETING_TOPIC', 'MEETING_DESCRIPTION',
                'START_TIME_UTC', 'END_TIME_UTC',
            ],
            'attendance': [
                'EVENTCODE', 'SESSION_ID', 'ATTENDEE_ID', 'ATTENDEDDATE', 'ATTENDEDTIME',
                'UTCATTENDEDTIME', 'ATTENDANCE_STATUS',
            ],
            'bookmarks': [
                'EVENTCODE', 'EMAIL', 'SESSION_ID', 'ATTENDEE_ID', 'SESSION_TIME',
                'SESSION_DATE', 'SESSION_TITLE', 'SESSION_TYPE', 'FIRST_NAME', 'LAST_NAME',
                'SCHEDULED_DATETIME', 'ATTENDED_DATETIME',
            ],
            'favorites': [],
        }

        self.recs_params = {
            'ret_recs_size': 30,
            'ret_recs_size_mult': 2,
            'session_sim_size': 1,
            'session_sim_mult': 0.01,
            'time_half_decay': 365,
            'attendance_type_weights': {
                'default': 0,
                'scheduled': 1,
                'waitlisted': 4,
                'attended': 5,
                'bookmark': 4,
            },
            'user_sim_size': 25,
            'norm_score': True,
            'popular_weight': 0.6,
            'nn_weight': 1,
            'emb_weight': 1,
            'popular_sessions_weights': {
                'Default': 1.0,
                'Main Keynote': 2,
                'Keynote': 1.8,
                'Theater': 1.2,
                'Breakout': 1.2,
            },
            'progress_every': 100,
        }

    def get_params(self):
        return deepcopy(self.recs_params)
