#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:05:22 2026

@author: mdraminski
"""

import pandas as pd
import numpy as np
import json
import logging
import math
from bisect import bisect_right
#from openpyxl as exl
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from embeddings import embeddings
from utils import clean_html, df_row_to_text

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)

###################################
def parse_times_column(df, times_col="TIMES", session_col="SESSION_ID"):
    def safe_int(value, default=0):
        if pd.isna(value) or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning("Could not cast value to int: %s", value)
            return default

    def empty_session_row(session_id, status):
        return {
            session_col: session_id,
            "slot_index": 1,
            "slot_count": 0,
            "capacity": 0,
            "registered": 0,
            "day": 0,
            "length": 0,
            "parse_status": status,
        }

    parsed_rows = []

    for _, row in df.iterrows():
        raw = row.get(times_col)
        session_id = row.get(session_col)

        raw_is_missing = raw is None or (
            not isinstance(raw, (list, dict)) and pd.isna(raw)
        )
        if raw_is_missing:
            parsed_rows.append(empty_session_row(session_id, "missing_times"))
            continue

        try:
            # Clean malformed JSON
            cleaned = raw if isinstance(raw, (list, dict)) else str(raw).replace('""', '"')
            
            # Parse JSON
            data = cleaned if isinstance(cleaned, (list, dict)) else json.loads(cleaned)

            if isinstance(data, dict):
                data = [data]

            if not data:
                parsed_rows.append(empty_session_row(session_id, "empty_times"))
                continue

            rows_before = len(parsed_rows)
            valid_items = [item for item in data if isinstance(item, dict)]
            slot_count = len(valid_items)

            for item in data:
                if not isinstance(item, dict):
                    logger.warning("Skipping non-object TIMES item SESSION_ID=%s: %s", session_id, item)
            
            for slot_index, item in enumerate(valid_items, start=1):
                item = item.copy()

                # Cast numeric fields safely
                item["capacity"] = safe_int(item.get("capacity", 0))
                item["registered"] = safe_int(item.get("registered", 0))
                item["day"] = safe_int(item.get("day", 0))
                item["length"] = safe_int(item.get("length", 0))
                item["slot_index"] = slot_index
                item["slot_count"] = slot_count
                item["parse_status"] = "parsed"

                # Attach SESSION_ID from original dataframe
                item_session_id = session_id if pd.notna(session_id) else item.get("sessionID")
                item[session_col] = item_session_id

                parsed_rows.append(item)

            if len(parsed_rows) == rows_before:
                parsed_rows.append(empty_session_row(session_id, "no_valid_times"))

        except Exception as e:
            logger.warning("Skipping row SESSION_ID=%s due to error: %s", session_id, e)
            parsed_rows.append(empty_session_row(session_id, "parse_error"))

    if not parsed_rows:
        return pd.DataFrame(columns=[session_col, "slot_index", "slot_count", "capacity", "registered", "day", "length", "parse_status"])

    #move session_col to 1st position
    parsed_rows = pd.DataFrame(parsed_rows)
    cols = list(parsed_rows.columns)
    cols.remove(session_col)
    cols = [session_col] + cols
    parsed_rows = parsed_rows[cols]

    return parsed_rows

###################################
def get_session_min_time(sessions, events):
    sessions_times = parse_times_column(sessions, 'TIMES', 'SESSION_ID')[['SESSION_ID','startTime']].rename(columns={'startTime':'SESSION_DATETIME'})
    sessions_times['SESSION_DATETIME'] = pd.to_datetime(sessions_times['SESSION_DATETIME'], errors='coerce')
    sessions_times = sessions_times.merge(sessions[['SESSION_ID','EVENTCODE']]).merge(events[['EVENTCODE','START_DATE']])
    mask = sessions_times["SESSION_DATETIME"].isna()
    sessions_times.loc[mask, "SESSION_DATETIME"] = pd.to_datetime(sessions_times.loc[mask, "START_DATE"].astype(str) + " 09:00:00", errors="coerce")
    sessions_times = sessions_times.groupby("SESSION_ID", as_index=False)["SESSION_DATETIME"].min()
    
    return sessions_times

###################################
def get_attendee(attendee, first_name, last_name):
    result = attendee.loc[
        (attendee['FIRST_NAME'].str.lower() == first_name.lower()) &
        (attendee['LAST_NAME'].str.lower() == last_name.lower())
    ]    
    return result

###################################
def calc_session_embeddings(sessions, speakers):
    sessions = sessions.copy()
    speakers = speakers.copy()
    #len(sessions)    
    #session_filter(sessions, True).to_csv(data_path + "sessions_filtered.csv", index = False)
    sessions['KEY_TAKEAWAYS_BY_EINSTEIN'] = sessions['KEY_TAKEAWAYS_BY_EINSTEIN'].apply(clean_html)
    #sessions.loc[~sessions['KEY_TAKEAWAYS_BY_EINSTEIN'].isna(),'KEY_TAKEAWAYS_BY_EINSTEIN']        
    logger.info("Sessions size: %s", len(sessions))
    
    #sessions['SESSION_SUMMARY_BY_EINSTEIN'].str.len().mean()
    #sessions['KEY_TAKEAWAYS_BY_EINSTEIN'].str.len().mean()
        
    col_map = {
            "TITLE": "Title",
            "CONTENT_SESSION_LEVEL": "Level",
            "ABSTRACT": "Abstract",
            "SESSION_SUMMARY_BY_EINSTEIN": "Summary",
            #"KEY_TAKEAWAYS_BY_EINSTEIN": "Summary",
            }
    sessions['SessionTXT'] = df_row_to_text(sessions, col_map)
    #sessions[['EVENTCODE','SESSION_ID','SessionTXT']]    
    logger.info("SessionTXT mean size: %s", int(sessions['SessionTXT'].str.len().mean()))
    
    col_map = {
            "GLOBAL_FULL_NAME": "Name",
            "GLOBAL_COMPANY": "Company",
            "GLOBAL_JOB_TITLE": "Job_Title",
            "BIO": "Bio"
            }
    speakers['SpeakerTXT'] = df_row_to_text(speakers, col_map)
    #speakers[['EVENTCODE','SESSION_ID','SpeakerTXT']]
    logger.info("SpeakerTXT mean size: %s", int(speakers['SpeakerTXT'].str.len().mean()))
    
    #you have two data.frames 
    #sessions[['EVENTCODE','SESSION_ID','SessionTXT']]
    #speakers[['EVENTCODE','SESSION_ID','SpeakerTXT']]
    #please join speakers to sessions by EVENTCODE and SESSION_ID but keep in mind that there can be more than one speaker per session. Therefore combine all speakers in such a way that create a new txt filed with Speaker1 ... Speaker2 and so one info and then join to sessions
    
    # Combine all speakers per session into one text field
    speakers_agg = (
        speakers.groupby(['EVENTCODE', 'SESSION_ID'], as_index=False)['SpeakerTXT']
        .apply(lambda s: ' | '.join(
            f"Speaker{i+1}: ({txt})" for i, txt in enumerate(s.dropna().astype(str))
        ))
        .rename(columns={'SpeakerTXT': 'SpeakersTXT'})
    )
    # Join to sessions
    sessions = sessions.merge(speakers_agg,on=['EVENTCODE', 'SESSION_ID'],how='left')
    
    # and create one full txt feature
    sessions['FullTXT'] = (
        sessions['SessionTXT'].fillna('') + ' | ' +  sessions['SpeakersTXT'].fillna('')
        ).str.strip(' |')
    #sessions[['EVENTCODE','SESSION_ID','FullTXT']]
    logger.info("FullTXT mean size: %s", int(sessions['FullTXT'].str.len().mean()))
            
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sessions['FullTXT'].to_list())
    
    return embeddings


###################################
def update_similarities(similarities, recs_params):    
    session_sim_size = recs_params.get('session_sim_size',1)
    session_sim_mult = recs_params.get('session_sim_mult',0.01)
        
    mask = similarities.rank(axis=1, ascending=False, method="first") <= session_sim_size
    similarities_new = similarities.where(mask, similarities * session_sim_mult)
    #df_new keeps the top n values in each row unchanged and multiplies all other row values by session_sim_mult.
    #To modify df in place:
    #similarities[:] = similarities.where(mask, df * session_sim_mult)
    
    return similarities_new    

###################################
def get_user_attendance(user_id, past_attendance, events = None):
    # case 1: user_id is DataFrame with ATTENDEE_ID + rating
    if isinstance(user_id, pd.DataFrame):
        user_df = user_id[['ATTENDEE_ID', 'rating']].copy()
    # case 2: user_id is list of attendee ids
    elif isinstance(user_id, list):
        user_df = pd.DataFrame({'ATTENDEE_ID': user_id})
        user_df['rating'] = 1.0
    # case 3: single attendee id
    else:
        user_df = pd.DataFrame({'ATTENDEE_ID': [user_id], 'rating': [1.0]})

    # attendance rows for selected users
    user_attendance = past_attendance.loc[past_attendance['ATTENDEE_ID'].isin(user_df['ATTENDEE_ID'])].copy()
    # attach rating from input
    user_attendance = user_attendance.merge(user_df.rename(columns={'rating':'user_rating'}) , how='left', on='ATTENDEE_ID')

    user_attendance['rating'] = user_attendance['rating'] * user_attendance['user_rating']    
    user_attendance = user_attendance.drop(columns=['user_rating'])
        
    # merge event info
    if events is not None:
        user_attendance = (user_attendance.merge(events[['EVENTCODE', 'START_DATE', 'NAME']], how='left', on='EVENTCODE')
                           .rename(columns={'START_DATE': 'EVENT_START_DATE', 'NAME': 'EVENT_NAME'})
                           .sort_values(by='EVENT_START_DATE'))

    # fallback when rating missing
    user_attendance['rating'] = user_attendance['rating'].fillna(1.0)

    return user_attendance

###################################
#user_id = user_nn_id[:user_nn_size]
#user_id = event_users_returning_id[1]
def calc_user_candidates(user_id, past_attendance, session_similarities, events = None, norm_score = True):    
    user_attendance = get_user_attendance(user_id, past_attendance, events = events)
    user_attendance = user_attendance[['ATTENDEE_ID','SESSION_ID','rating']]
                
    user_similarities = session_similarities[session_similarities.index.isin(set(user_attendance['SESSION_ID']))]
    user_similarities = user_similarities.stack().reset_index()
    user_similarities.columns = ["SESSIONIN", "SESSIONOUT", "score"]
    if(len(user_similarities)==0):
        return None
    #user_similarities
    
    #len(set(user_similarities['SESSIONIN']))
    #len(set(user_similarities['SESSIONOUT']))
    user_attendance = user_attendance.merge(user_similarities.rename(columns={'SESSIONIN':'SESSION_ID'}), how = 'left', on='SESSION_ID')
    user_attendance['score'] = user_attendance['rating'] * user_attendance['score']
    ret_recs = user_attendance.groupby(['SESSIONOUT'])['score'].sum().reset_index().sort_values('score',ascending= False).rename(columns={'SESSIONOUT':'SESSION_ID'})
    if norm_score:
        ret_recs['score'] = ret_recs['score']/max(ret_recs['score'])
    
    # case 1: user_id is DataFrame with ATTENDEE_ID + rating
    if isinstance(user_id, pd.DataFrame):
        ret_recs['ATTENDEE_ID'] = ""
    # case 2: user_id is list of attendee ids
    elif isinstance(user_id, list):
        ret_recs['ATTENDEE_ID'] = ""
    # case 3: single attendee id
    else:
        ret_recs['ATTENDEE_ID'] = user_id
    
    return ret_recs[['ATTENDEE_ID','SESSION_ID','score']]

###################################
#user_id = user_nn_id[:user_nn_size]
#user_id = event_users_returning_id[1]
#session_embeddings_df = session_embeddings.get_dataframe()
def calc_user_embedding(user_id, past_attendance, session_embeddings_df):
    from embeddings import embeddings

    if not isinstance(session_embeddings_df, pd.DataFrame):
        if hasattr(session_embeddings_df, "get_dataframe"):
            session_embeddings_df = session_embeddings_df.get_dataframe()
        else:
            raise TypeError("session_embeddings_df must be a DataFrame or embeddings object")

    user_attendance = get_user_attendance(user_id, past_attendance, events = None)
    user_attendance = user_attendance[['ATTENDEE_ID','SESSION_ID','rating']]                   
    #len(set(user_similarities['SESSIONOUT']))
    user_attendance = user_attendance.merge(session_embeddings_df, how = 'left', on='SESSION_ID')    
    user_attendance['rating'] = user_attendance['rating'] / sum(user_attendance['rating'])
    emb_cols = user_attendance.columns[user_attendance.columns.get_loc("rating") + 1:]    
    user_attendance[emb_cols] = user_attendance[emb_cols].mul(user_attendance["rating"], axis=0)    
    user_embedding = user_attendance[emb_cols].to_numpy().sum(axis=0)    
    user_embedding = embeddings(user_embedding, user_id)
    
    return user_embedding

###################################
def session_filter(df, keep=True, version=2):
    """Filter sessions to the recommendable set.

    The filtering logic lives in sfloader._session_filter; this thin public
    delegate is kept so callers using `sutil.session_filter(...)` resolve
    (the symbol had moved onto the sfloader class). Lazy imports avoid any
    import cycle (sfloader/sfconfig do not import sfutils).
    """
    from sfconfig import sfconfig
    from sfloader import sfloader
    return sfloader(sfconfig())._session_filter(df, keep=keep, version=version)


###################################
def get_sessions_filter(attendee, sessions):
    ret_filter = []
    #GA users filter
    #attendee['ATTENDEE_TYPE'].value_counts()[:30]
    attendee_filter = list(set(attendee.loc[attendee['ATTENDEE_TYPE'] == 'Full Conference','ATTENDEE_ID']))
    sessions_filter = list(set(sessions.loc[sessions['SESSION_TYPE'].isin(['Executive Summit']),'SESSION_ID']))
    ret_filter_dict = {'name': 'GA_users_filter', 'users':attendee_filter, 'sessions':sessions_filter}
    ret_filter.append(ret_filter_dict)
    
    return ret_filter

###################################
#session_similarities = event_session_similarities
def calc_users_recs(user_ids, past_attendance, session_similarities, user_sessions_filter = None, events=None, recs_params=None):
    logger.info("Users to process: %s", len(user_ids))

    ret_recs_size = recs_params.get("ret_recs_size", 30)
    progress_every = recs_params.get("progress_every", 100)

    ret_recs = []
    for i, user_id in enumerate(user_ids):
        if progress_every and len(user_ids) >= progress_every and i % progress_every == 0:
            logger.info("Regular recommendations processed: %s/%s", i, len(user_ids))

        user_candidates_recs = calc_user_candidates(user_id, past_attendance, session_similarities, events)
        if user_sessions_filter is not None:
            for sfilter in user_sessions_filter:
                if user_id in sfilter['users']:
                    user_candidates_recs = user_candidates_recs.loc[~user_candidates_recs['SESSION_ID'].isin(sfilter['sessions'])]
        
        if user_candidates_recs is not None and not user_candidates_recs.empty:
            ret_recs.append(user_candidates_recs[0:ret_recs_size])
        else:
            logger.warning("Empty result for user: %s", user_id)

    if not ret_recs:
        logger.warning("No recommendations generated.")
        return pd.DataFrame(columns=['ATTENDEE_ID', 'SESSION_ID', 'score'])

    ret_recs = pd.concat(ret_recs, ignore_index=True)
    logger.info("Users in ret_recs: %s", len(set(ret_recs['ATTENDEE_ID'])))

    return ret_recs

###################################
#session_similarities = event_session_similarities
def calc_users_embeddings(user_ids, past_attendance, session_embeddings, progress_every=100):
    #from embeddings import embeddings
    
    if isinstance(user_ids, (str, bytes)):
        user_ids = [user_ids]
    else:
        user_ids = list(user_ids)

    logger.info("Users to process: %s", len(user_ids))
    if isinstance(session_embeddings, pd.DataFrame):
        session_embeddings_df = session_embeddings
    elif hasattr(session_embeddings, "get_dataframe"):
        session_embeddings_df = session_embeddings.get_dataframe()
    else:
        raise TypeError("session_embeddings must be a DataFrame or embeddings object")

    users_embeddings_data = []
    users_rownames = []
    for i, user_id in enumerate(user_ids):
        if progress_every and len(user_ids) >= progress_every and i % progress_every == 0:
            logger.info("Calculation of users embeddings processed: %s/%s", i, len(user_ids))
        
        user_embedding = calc_user_embedding(user_id, past_attendance, session_embeddings_df)
        users_embeddings_data.append(user_embedding.embeddings[0])
        users_rownames.append(user_embedding.rownames[0])
            
    if not users_embeddings_data:
        session_id_col_idx = session_embeddings_df.columns.get_loc("SESSION_ID")
        embedding_cols = session_embeddings_df.columns[session_id_col_idx + 1:]
        return embeddings(np.empty((0, len(embedding_cols))), [])

    return embeddings(np.vstack(users_embeddings_data), users_rownames)

###################################
def calc_user_similarity_from_attendance(past_attendance, top_n=10, batch_size=1000, attendee_col='ATTENDEE_ID',session_col='SESSION_ID'):
    required_cols = [attendee_col, session_col]
    missing_cols = [col for col in required_cols if col not in past_attendance.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    past_attendance = past_attendance.dropna(subset=required_cols)
    if past_attendance.empty:
        return pd.DataFrame(columns=['ATTENDEE_IN', 'ATTENDEE_OUT', 'score'])

    attendee_codes, attendee_index = pd.factorize(past_attendance[attendee_col])
    session_codes, session_index = pd.factorize(past_attendance[session_col])
    data = np.ones(len(past_attendance), dtype=np.float32)
    data = coo_matrix(
        (data, (attendee_codes, session_codes)),
        shape=(len(attendee_index), len(session_index)),
    ).tocsr()

    sim_users = cosine_similarity(data)
    np.fill_diagonal(sim_users, 0)
    sim_users = pd.DataFrame(sim_users, index=attendee_index, columns=attendee_index)
    return matrix2long(
        sim_users,
        top_n=top_n,
        batch_size=batch_size,
        new_cols=['ATTENDEE_IN', 'ATTENDEE_OUT', 'score'],
    )

###################################
def filter_user_candidates(user_candidates_recs, user_id, user_sessions_filter):
    if user_sessions_filter is not None:
        for sfilter in user_sessions_filter:
            if user_id in sfilter['users']:
                user_candidates_recs = user_candidates_recs.loc[~user_candidates_recs['SESSION_ID'].isin(sfilter['sessions'])]
                
    return user_candidates_recs

###################################
#user_id = cold_start_users[602]
#pd.concat([event_real_attendee.loc[event_real_attendee['ATTENDEE_ID'] == user_id,'description'], user_nn_id.merge(event_real_attendee[['ATTENDEE_ID','description']],how='left',on='ATTENDEE_ID')]).to_csv('~/user_nn_id_13.csv')
def calc_cold_start_recs_nn(user_ids, past_attendance, user_similarities, session_similarities, user_sessions_filter, recs_params, ret_recs_size=30):
    user_sim_size = recs_params['user_sim_size']
    norm_score = recs_params['norm_score']
    progress_every = recs_params['progress_every']
    
    user_ids = (np.intersect1d(user_ids, user_similarities['ATTENDEE_IN'].to_list())).tolist()
    logger.info("Cold-start users to process: %s", len(user_ids))

    ret_recs = []
    for i, user_id in enumerate(user_ids):
        if progress_every and len(user_ids) >= progress_every and i % progress_every == 0:
            logger.info("Cold-start recommendations processed: %s/%s", i, len(user_ids))

        user_nn_id = (user_similarities.loc[user_similarities['ATTENDEE_IN'] == user_id, ['ATTENDEE_OUT', 'score']]
                      .rename(columns={'ATTENDEE_OUT': 'ATTENDEE_ID', 'score': 'rating'})
                      .sort_values(by='rating', ascending=False))
        
        user_candidates_recs = calc_user_candidates(user_nn_id[:user_sim_size], past_attendance, session_similarities, None, norm_score)
        if user_candidates_recs is not None and not user_candidates_recs.empty:
            user_candidates_recs['ATTENDEE_ID'] = user_id            
            user_candidates_recs = filter_user_candidates(user_candidates_recs, user_id, user_sessions_filter)
            ret_recs.append(user_candidates_recs[0:ret_recs_size])
        else:
            logger.warning("Empty result for user: %s", user_id)

    if not ret_recs:
        logger.warning("No cold-start recommendations generated.")
        return pd.DataFrame(columns=['ATTENDEE_ID', 'SESSION_ID', 'score'])

    ret_recs = pd.concat(ret_recs, ignore_index=True)
    logger.info("Cold-start users in ret_recs: %s", len(set(ret_recs['ATTENDEE_ID'])))

    return ret_recs

###################################
def calc_cold_start_recs_emb(user_ids, user_embeddings, session_embeddings, user_sessions_filter, recs_params, ret_recs_size=30):
    norm_score = recs_params['norm_score']
    progress_every = recs_params['progress_every']
    
    user_ids = (np.intersect1d(user_ids, user_embeddings.rownames)).tolist()
    logger.info("Cold-start users to process: %s", len(user_ids))

    ret_recs = []
    for i, user_id in enumerate(user_ids):
        if progress_every and len(user_ids) >= progress_every and i % progress_every == 0:
            logger.info("Cold-start recommendations processed: %s/%s", i, len(user_ids))

        user_candidates_recs = session_embeddings.calc_cosine_similarities(user_embeddings.get(user_id), True)
        user_candidates_recs = user_candidates_recs.rename(columns={"rownames": "SESSION_ID", "similarity": "score"})
        user_candidates_recs["score"] = (user_candidates_recs["score"] + 1) / 2
        user_candidates_recs["ATTENDEE_ID"] = user_id
        user_candidates_recs = filter_user_candidates(user_candidates_recs, user_id, user_sessions_filter)
        user_candidates_recs = user_candidates_recs.head(ret_recs_size).copy()        
        if norm_score:
            user_candidates_recs["score"] = user_candidates_recs["score"] / user_candidates_recs["score"].max()
        ret_recs.append(user_candidates_recs[["ATTENDEE_ID", "SESSION_ID", "score"]])
        
    if not ret_recs:
        logger.warning("No cold-start recommendations generated.")
        return pd.DataFrame(columns=['ATTENDEE_ID', 'SESSION_ID', 'score'])

    ret_recs = pd.concat(ret_recs, ignore_index=True)
    logger.info("Cold-start users in ret_recs: %s", len(set(ret_recs['ATTENDEE_ID'])))

    return ret_recs[["ATTENDEE_ID", "SESSION_ID", "score"]]

###################################
def get_popular_sessions(event_sessions, session_room, user_sessions_filter, recs_params, ret_recs_size=30):
    session_avg_capacity = session_room.groupby(["SESSION_ID"], as_index=False).agg(avg_capacity=("CAPACITY", "mean")).sort_values("avg_capacity", ascending=False).reset_index(drop=True)
    session_avg_capacity["avg_capacity"] = session_avg_capacity["avg_capacity"].round(0).astype(int)    
    session_avg_capacity = session_avg_capacity.merge(event_sessions[['SESSION_ID','SESSION_TYPE']], on='SESSION_ID', how='left')
    pop_weights = recs_params['popular_sessions_weights']    
    session_avg_capacity['weight'] = pop_weights['Default']
    session_avg_capacity["weight"] = session_avg_capacity["weight"].astype(float)
    for w in pop_weights.keys():
        session_avg_capacity.loc[session_avg_capacity['SESSION_TYPE'] == w, 'weight'] = pop_weights[w]
        
    #session_avg_capacity['avg_capacity'] = session_avg_capacity['avg_capacity'] * session_avg_capacity['weight']    
    session_avg_capacity["score"] = np.log10(session_avg_capacity["avg_capacity"] + 1)
    #session_avg_capacity["score"] = np.log(session_avg_capacity["avg_capacity"] + 1)    
    session_avg_capacity["score"] = session_avg_capacity["score"] / session_avg_capacity["score"].max()
    session_avg_capacity['score'] = session_avg_capacity['score'] * session_avg_capacity['weight']
    session_avg_capacity["score"] = session_avg_capacity["score"] / session_avg_capacity["score"].max()    
    session_avg_capacity = session_avg_capacity.sort_values("score", ascending=False)    
    #session_avg_capacity[:50]
    
    session_avg_capacity = session_avg_capacity[['SESSION_ID','score']]

    return session_avg_capacity[0:ret_recs_size]

###################################
def get_recs_popular_for_users(user_ids, ret_recs_popular, user_sessions_filter, ret_recs_size=30):
    ret_recs = []
    for user_id in user_ids:
        user_recs = ret_recs_popular.copy()
        user_recs["ATTENDEE_ID"] = user_id
        user_recs = user_recs[["ATTENDEE_ID", "SESSION_ID", "score"]]        
        user_recs = filter_user_candidates(user_recs, user_id, user_sessions_filter)
        ret_recs.append(user_recs[0:ret_recs_size])

    return pd.concat(ret_recs, ignore_index=True)

###################################
def calc_combined_cold_start_recs(user_id, user_similarities, session_similarities, user_embeddings, event_embeddings, past_attendance, event_sessions, session_room, user_sessions_filter, recs_params):
    
    ret_recs_size = recs_params.get("ret_recs_size",30)
    ret_recs_size_mult = recs_params.get("ret_recs_size_mult",2)
    popular_weight = recs_params.get("popular_weight",0.6)
    nn_weight = recs_params.get("nn_weight",1)
    emb_weight = recs_params.get("emb_weight",1)
    
    ret_recs_popular = get_popular_sessions(event_sessions, session_room, user_sessions_filter, recs_params, ret_recs_size * ret_recs_size_mult).copy()
    user_ret_recs_popular = get_recs_popular_for_users([user_id], ret_recs_popular, user_sessions_filter, ret_recs_size * ret_recs_size_mult)
    user_ret_recs_nn = calc_cold_start_recs_nn([user_id], past_attendance, user_similarities, session_similarities, user_sessions_filter, recs_params, ret_recs_size * ret_recs_size_mult).copy()
    user_ret_recs_emb = calc_cold_start_recs_emb([user_id], user_embeddings, event_embeddings, user_sessions_filter, recs_params, ret_recs_size * ret_recs_size_mult).copy()

    user_ret_recs_popular["weight"] = popular_weight
    user_ret_recs_nn["weight"] = nn_weight
    user_ret_recs_emb["weight"] = emb_weight

    user_ret_recs = pd.concat([user_ret_recs_nn, user_ret_recs_emb, user_ret_recs_popular], ignore_index=True)
    user_ret_recs["score"] = user_ret_recs["score"] * user_ret_recs["weight"]
    user_ret_recs = user_ret_recs.groupby("SESSION_ID", as_index=False)["score"].sum()
    user_ret_recs = user_ret_recs.sort_values("score", ascending=False)

    max_score = user_ret_recs["score"].max()
    if pd.notna(max_score) and max_score != 0:
        user_ret_recs["score"] = user_ret_recs["score"] / max_score

    return user_ret_recs[0:ret_recs_size]

###################################
def get_nn_users(user_id, user_similarities, event_attendee):
    user_sim = user_similarities.loc[user_similarities['ATTENDEE_IN']==user_id]
    _self_desc = event_attendee.loc[event_attendee['ATTENDEE_ID']==user_id,'description']
    user_sim.loc[len(user_sim)] = {
        "ATTENDEE_IN": user_id, "ATTENDEE_OUT": user_id, "score": 1.0,
        "description": _self_desc.iloc[0] if not _self_desc.empty else ""}
    user_sim = user_sim.sort_values('score', ascending = False).reset_index(drop=True)
    user_sim = user_sim.merge(event_attendee[['ATTENDEE_ID','description']] , how='left', left_on = 'ATTENDEE_OUT', right_on='ATTENDEE_ID')
    
    return user_sim

###################################
def show_nice_recs(ret_recs, event_sessions):
    return ret_recs.merge(event_sessions[['SESSION_ID','TITLE','SESSION_TYPE','LENGTH','TIMES_OFFERED','SESSION_TRACK']], on='SESSION_ID', how='left')

###################################
def optimize_session_schedule(
    session_room_user,
    start_time=None,
    session_col='SESSION_ID',
    start_col='START_TIME',
    end_col='END_TIME',
    score_col='score',
    length_col='LENGTH',
    break_minutes=0,
    short_session_penalty=0.0,
    optimization_score_col='schedule_score',
    dedupe_sessions=True,
):
    """Select a non-overlapping schedule that maximizes total score.

    When dedupe_sessions=True, the function first keeps one candidate row per
    SESSION_ID using highest score, then earliest start/end as tie-breakers.
    It then uses weighted interval scheduling to maximize the score across the
    full schedule.

    Args:
        break_minutes: Required gap between one session's END_TIME and the next
            session's START_TIME.
        short_session_penalty: Value from 0 to 1. A value of 0 optimizes only
            by score. Values between 0 and 1 increasingly reduce scores for
            shorter sessions. A value of 1 is a hard penalty: only sessions
            with the maximum candidate length keep their score.
    """
    required_cols = [session_col, start_col, end_col, score_col]
    missing_cols = [col for col in required_cols if col not in session_room_user.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if break_minutes < 0:
        raise ValueError("break_minutes must be >= 0")
    if short_session_penalty < 0 or short_session_penalty > 1:
        raise ValueError("short_session_penalty must be between 0 and 1")

    candidates = session_room_user.copy()
    candidates[start_col] = pd.to_datetime(candidates[start_col], errors='coerce')
    candidates[end_col] = pd.to_datetime(candidates[end_col], errors='coerce')
    candidates[score_col] = pd.to_numeric(candidates[score_col], errors='coerce')

    candidates = candidates.dropna(subset=[session_col, start_col, end_col, score_col])
    candidates = candidates.loc[candidates[end_col] > candidates[start_col]].copy()

    if start_time is not None:
        start_time = pd.to_datetime(start_time)
        candidates = candidates.loc[candidates[start_col] >= start_time].copy()

    if candidates.empty:
        logger.warning("No valid session candidates available for scheduling.")
        return candidates

    duration_minutes = (candidates[end_col] - candidates[start_col]).dt.total_seconds() / 60
    if length_col in candidates.columns:
        candidate_lengths = pd.to_numeric(candidates[length_col], errors='coerce')
        candidates['_schedule_length_minutes'] = candidate_lengths.fillna(duration_minutes)
    else:
        candidates['_schedule_length_minutes'] = duration_minutes
    candidates['_schedule_length_minutes'] = candidates['_schedule_length_minutes'].clip(lower=0)

    max_length = candidates['_schedule_length_minutes'].max()
    candidates['_schedule_length_factor'] = 1.0
    if max_length > 0:
        candidates['_schedule_length_factor'] = candidates['_schedule_length_minutes'] / max_length

    candidates[optimization_score_col] = candidates[score_col]
    if short_session_penalty > 0:
        if short_session_penalty == 1:
            length_weight = np.isclose(candidates['_schedule_length_factor'], 1.0).astype(float)
        else:
            penalty_power = short_session_penalty / (1 - short_session_penalty)
            length_weight = candidates['_schedule_length_factor'] ** penalty_power

        candidates[optimization_score_col] = candidates[score_col] * length_weight

    if dedupe_sessions:
        candidates = (
            candidates
            .sort_values(
                [session_col, score_col, optimization_score_col, start_col, end_col],
                ascending=[True, False, False, True, True],
            )
            .drop_duplicates(subset=[session_col], keep='first')
            .copy()
        )

    candidates = (
        candidates
        .sort_values([end_col, start_col, score_col], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    starts = candidates[start_col].tolist()
    ends_with_break = (
        candidates[end_col] + pd.to_timedelta(break_minutes, unit='m')
    ).tolist()
    scores = candidates[optimization_score_col].tolist()

    # previous_non_overlap[i] points to the latest row ending early enough to
    # preserve break_minutes before row i starts.
    previous_non_overlap = [
        bisect_right(ends_with_break, starts[i]) - 1
        for i in range(len(candidates))
    ]

    best_score = [0.0] * (len(candidates) + 1)
    take_row = [False] * len(candidates)

    for i in range(1, len(candidates) + 1):
        row_idx = i - 1
        score_with_row = scores[row_idx] + best_score[previous_non_overlap[row_idx] + 1]
        score_without_row = best_score[i - 1]

        if score_with_row > score_without_row:
            best_score[i] = score_with_row
            take_row[row_idx] = True
        else:
            best_score[i] = score_without_row

    selected_indices = []
    i = len(candidates)
    while i > 0:
        row_idx = i - 1
        score_with_row = scores[row_idx] + best_score[previous_non_overlap[row_idx] + 1]

        if take_row[row_idx] and score_with_row >= best_score[i - 1]:
            selected_indices.append(row_idx)
            i = previous_non_overlap[row_idx] + 1
        else:
            i -= 1

    schedule = candidates.iloc[list(reversed(selected_indices))].copy()
    schedule = schedule.sort_values(start_col).reset_index(drop=True)
    schedule['schedule_rank'] = range(1, len(schedule) + 1)
    schedule['score_rank'] = schedule[score_col].rank(
        method='first',
        ascending=False,
    ).astype(int)
    schedule['schedule_score_rank'] = schedule[optimization_score_col].rank(
        method='first',
        ascending=False,
    ).astype(int)

    logger.info(
        "Selected %s sessions with total score %.4f and schedule score %.4f",
        len(schedule),
        schedule[score_col].sum(),
        schedule[optimization_score_col].sum(),
    )

    hidden_output_cols = [
        '_schedule_length_minutes',
        '_schedule_length_factor',
        'schedule_score_rank',
        'schedule_rank',
    ]
    if optimization_score_col != score_col:
        hidden_output_cols.append(optimization_score_col)

    schedule = schedule.drop(columns=hidden_output_cols, errors='ignore')
    return schedule

###################################
def build_attendee_description(x):
    parts = []
    #cols = ["FIRST_NAME", "LAST_NAME", "COMPANY_NAME","JOB_TITLE", "PRIMARY_ROLE", "PRIMARY_INDUSTRY"]
    #x[cols] = x[cols].fillna("")

    name = " ".join(filter(None, [x["FIRST_NAME"], x["LAST_NAME"]])).strip()
    if name:
        parts.append(name)

    if x["COMPANY_NAME"]:
        parts.append(f" works at {x['COMPANY_NAME']}")

    if x["JOB_TITLE"]:
        parts.append(f"as {x['JOB_TITLE']}")

    if x["PRIMARY_ROLE"]:
        parts.append(f"and his/her primary role is {x['PRIMARY_ROLE']}")

    if x["PRIMARY_INDUSTRY"]:
        parts.append(f"and the company industry is {x['PRIMARY_INDUSTRY']}")

    return " ".join(parts)

###################################
def matrix2long(df, top_n=10, batch_size=1000, new_cols = ['row', 'col', 'value'] ):
    if df.empty:
        return pd.DataFrame(columns=new_cols)

    top_n = min(top_n, df.shape[1])
    if top_n <= 0:
        return pd.DataFrame(columns=new_cols)

    results = []

    values = df.values
    index = df.index.to_numpy()
    columns = df.columns.to_numpy()

    n_rows = df.shape[0]

    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        logger.info("Processed rows up to %s", end)
        batch_vals = values[start:end]
        batch_index = index[start:end]

        # Get indices of top_n elements per row (fast, no full sort)
        top_idx = np.argpartition(-batch_vals, top_n - 1, axis=1)[:, :top_n]

        # Get corresponding scores
        top_scores = np.take_along_axis(batch_vals, top_idx, axis=1)

        # Optional: sort top_n within each row
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)

        # Build output rows
        for i in range(end - start):
            attendee_in = batch_index[i]
            cols = columns[top_idx[i]]
            scores = top_scores[i]

            results.append(pd.DataFrame({
                new_cols[0]: attendee_in,
                new_cols[1]: cols,
                new_cols[2]: scores
            }))

    return pd.concat(results, ignore_index=True)

###################################
def calc_attendee_similarity(event_users_returning_id, event_users_coldstart_id, attendee_embeddings, sentenceTransformerModel, cold_start_batch_size=1000, top_n=25, matrix_batch_size=500, progress_every=1):
    if cold_start_batch_size <= 0:
        raise ValueError("cold_start_batch_size must be greater than 0")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0")

    def unique_list(values):
        if isinstance(values, (str, bytes)):
            values = [values]
        return list(dict.fromkeys(list(values)))

    event_users_returning_id = unique_list(event_users_returning_id)
    event_users_coldstart_id = unique_list(event_users_coldstart_id)

    logger.info("Returning users: %s", len(event_users_returning_id))
    logger.info("Cold-start users: %s", len(event_users_coldstart_id))

    ret_similarity = []
    total_cold_start = len(event_users_coldstart_id)

    for start in range(0, total_cold_start, cold_start_batch_size):
        end = min(start + cold_start_batch_size, total_cold_start)
        batch_no = start // cold_start_batch_size + 1
        coldstart_batch_id = event_users_coldstart_id[start:end]
        batch_users_id = unique_list(event_users_returning_id + coldstart_batch_id)

        if progress_every and batch_no % progress_every == 0:
            logger.info("Calculating attendee similarity for cold-start users %s/%s", end, total_cold_start)

        attendee_embeddings_batch = attendee_embeddings.select(batch_users_id)
        if attendee_embeddings_batch.embeddings.shape[0] == 0:
            continue

        attendee_similarity = sentenceTransformerModel.similarity(
            attendee_embeddings_batch.embeddings,
            attendee_embeddings_batch.embeddings,
        ).numpy()
        np.fill_diagonal(attendee_similarity, 0)

        attendee_similarity = pd.DataFrame(
            attendee_similarity,
            index=attendee_embeddings_batch.rownames,
            columns=attendee_embeddings_batch.rownames,
        )
        attendee_similarity.index.name = attendee_similarity.columns.name = None

        if start == 0:
            row_ids = attendee_similarity.index.tolist()
        else:
            row_ids = [user_id for user_id in coldstart_batch_id if user_id in attendee_similarity.index]
        col_ids = [user_id for user_id in event_users_returning_id if user_id in attendee_similarity.columns]

        if not row_ids or not col_ids:
            continue

        attendee_similarity = attendee_similarity.loc[row_ids, col_ids]
        attendee_similarity = matrix2long(
            attendee_similarity,
            top_n=top_n,
            batch_size=matrix_batch_size,
            new_cols=["ATTENDEE_IN", "ATTENDEE_OUT", "score"],
        )
        ret_similarity.append(attendee_similarity)

    if not ret_similarity:
        return pd.DataFrame(columns=["ATTENDEE_IN", "ATTENDEE_OUT", "score"])

    return pd.concat(ret_similarity, ignore_index=True)

###################################
def calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30]):
    attendee_hit_ratio_list = []
    session_hit_ratio_list = []
    for current_top_n in recs_top_n_list:
        logger.info("Calculating stats for top_n=%s", current_top_n)
        ret_recs_test = ret_recs.sort_values(["ATTENDEE_ID", "score"], ascending=[True, False]).groupby("ATTENDEE_ID").head(current_top_n)        
        recs_size = len(ret_recs_test)        
        ret_recs_test = ret_recs_test.merge(event_attendance, how = 'left', on = ['SESSION_ID', 'ATTENDEE_ID'])
        ret_recs_hit = ret_recs_test.loc[~ret_recs_test['ATTENDANCE_STATUS'].isna(),]
                                
        session_hit_ratio_list.append({
            "top_n": current_top_n,
            "recs_size": recs_size,
            "recs_hit": len(ret_recs_hit),
            "recs_hit_ratio": len(ret_recs_hit) / recs_size if recs_size > 0 else 0,
            "recs_hit_rating_sum": ret_recs_hit["rating"].sum(),
            "recs_hit_rating_avg": ret_recs_hit["rating"].mean() if len(ret_recs_hit) > 0 else 0,
        })               
        
        attendee_hit_cnt = pd.DataFrame(ret_recs_hit.value_counts('ATTENDEE_ID')).reset_index()
        attendee_hit_cnt = pd.DataFrame(attendee_hit_cnt['count'].value_counts()).rename(columns={'count':'cnt'}).reset_index().rename(columns={'count':'hits'}).sort_values('hits')        
        if current_top_n == recs_top_n:
            attendee_hit_cnt_result = attendee_hit_cnt.copy()        
        #attendee_hit_cnt.to_csv('~/b.csv')        
        attendee_hit_ratio = len(set(ret_recs_hit['ATTENDEE_ID']))/len(set(ret_recs['ATTENDEE_ID']))        
        attendee_hit_ratio_list.append(attendee_hit_ratio)

    session_hit_ratio_result = pd.DataFrame(session_hit_ratio_list)
    attendee_hit_ratio_result = pd.DataFrame({
            'top_n_recs':recs_top_n_list,
            'attendees_pct':np.round(np.array(attendee_hit_ratio_list), decimals=4)})
    return attendee_hit_ratio_result, attendee_hit_cnt_result, session_hit_ratio_result

###################################
def calc_ndcg(ret_recs, event_attendance, recs_top_n=15, return_details=False):
    required_recs_cols = ['ATTENDEE_ID', 'SESSION_ID', 'score']
    required_attendance_cols = ['ATTENDEE_ID', 'SESSION_ID', 'rating']
    missing_recs_cols = [col for col in required_recs_cols if col not in ret_recs.columns]
    missing_attendance_cols = [col for col in required_attendance_cols if col not in event_attendance.columns]

    if missing_recs_cols:
        raise ValueError(f"ret_recs is missing required columns: {missing_recs_cols}")
    if missing_attendance_cols:
        raise ValueError(f"event_attendance is missing required columns: {missing_attendance_cols}")
    if recs_top_n <= 0:
        raise ValueError("recs_top_n must be greater than 0")

    recs = ret_recs[required_recs_cols].copy()
    recs['score'] = pd.to_numeric(recs['score'], errors='coerce')
    recs = recs.dropna(subset=['ATTENDEE_ID', 'SESSION_ID', 'score'])

    if recs.empty:
        if return_details:
            return 0.0, pd.DataFrame(columns=['ATTENDEE_ID', 'dcg', 'idcg', 'ndcg'])
        return 0.0

    attendance = event_attendance[required_attendance_cols].copy()
    attendance['rating'] = pd.to_numeric(attendance['rating'], errors='coerce').fillna(1.0)
    attendance = attendance.dropna(subset=['ATTENDEE_ID', 'SESSION_ID'])

    recs = recs.groupby(['ATTENDEE_ID', 'SESSION_ID'], as_index=False)['score'].max()
    relevance = attendance.groupby(['ATTENDEE_ID', 'SESSION_ID'], as_index=False)['rating'].max()

    recs_top = (
        recs.sort_values(['ATTENDEE_ID', 'score', 'SESSION_ID'], ascending=[True, False, True])
        .groupby('ATTENDEE_ID', sort=False)
        .head(recs_top_n)
        .copy()
    )
    recs_top['rank'] = recs_top.groupby('ATTENDEE_ID').cumcount() + 1
    recs_top = recs_top.merge(
        relevance.rename(columns={'rating': 'relevance'}),
        how='left',
        on=['ATTENDEE_ID', 'SESSION_ID'],
    )
    recs_top['relevance'] = recs_top['relevance'].fillna(0.0)
    recs_top['dcg_value'] = (
        (np.power(2.0, recs_top['relevance']) - 1.0)
        / np.log2(recs_top['rank'] + 1.0)
    )
    user_dcg = recs_top.groupby('ATTENDEE_ID')['dcg_value'].sum().rename('dcg')

    user_ids = pd.Index(recs['ATTENDEE_ID'].drop_duplicates(), name='ATTENDEE_ID')
    ideal_top = (
        relevance.loc[relevance['ATTENDEE_ID'].isin(user_ids)]
        .sort_values(['ATTENDEE_ID', 'rating', 'SESSION_ID'], ascending=[True, False, True])
        .groupby('ATTENDEE_ID', sort=False)
        .head(recs_top_n)
        .copy()
    )
    ideal_top['rank'] = ideal_top.groupby('ATTENDEE_ID').cumcount() + 1
    ideal_top['idcg_value'] = (
        (np.power(2.0, ideal_top['rating']) - 1.0)
        / np.log2(ideal_top['rank'] + 1.0)
    )
    user_idcg = ideal_top.groupby('ATTENDEE_ID')['idcg_value'].sum().rename('idcg')

    user_ndcg = pd.DataFrame(index=user_ids).join(user_dcg).join(user_idcg).fillna(0.0)
    user_ndcg['ndcg'] = np.divide(
        user_ndcg['dcg'],
        user_ndcg['idcg'],
        out=np.zeros(len(user_ndcg), dtype=float),
        where=user_ndcg['idcg'].to_numpy() != 0,
    )
    user_ndcg = user_ndcg.reset_index()
    mean_ndcg = float(user_ndcg['ndcg'].mean()) if not user_ndcg.empty else 0.0

    if return_details:
        return mean_ndcg, user_ndcg
    return mean_ndcg

###################################
def compare_similar_users(similar_users_a, similar_users_b, top_n=None, return_details=False,
                          user_col='ATTENDEE_IN', similar_user_col='ATTENDEE_OUT', score_col='score'):
    required_cols = [user_col, similar_user_col, score_col]
    missing_a_cols = [col for col in required_cols if col not in similar_users_a.columns]
    missing_b_cols = [col for col in required_cols if col not in similar_users_b.columns]

    if missing_a_cols:
        raise ValueError(f"similar_users_a is missing required columns: {missing_a_cols}")
    if missing_b_cols:
        raise ValueError(f"similar_users_b is missing required columns: {missing_b_cols}")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be greater than 0 or None")

    def prepare_top_users(similar_users, source_name):
        ret = similar_users[required_cols].copy()
        ret[score_col] = pd.to_numeric(ret[score_col], errors='coerce')
        ret = ret.dropna(subset=[user_col, similar_user_col, score_col])
        ret = ret.groupby([user_col, similar_user_col], as_index=False)[score_col].max()
        ret = ret.sort_values([user_col, score_col, similar_user_col], ascending=[True, False, True])

        if top_n is not None:
            ret = ret.groupby(user_col, sort=False).head(top_n).copy()

        ret[f'{source_name}_rank'] = ret.groupby(user_col).cumcount() + 1
        return ret.rename(columns={score_col: f'{source_name}_score'})

    users_a = prepare_top_users(similar_users_a, 'a')
    users_b = prepare_top_users(similar_users_b, 'b')

    user_ids = pd.Index(
        pd.concat([users_a[user_col], users_b[user_col]], ignore_index=True).drop_duplicates(),
        name=user_col,
    )

    a_count = users_a.groupby(user_col).size().rename('a_count')
    b_count = users_b.groupby(user_col).size().rename('b_count')

    overlap_pairs = users_a.merge(
        users_b,
        how='inner',
        on=[user_col, similar_user_col],
    )
    overlap_count = overlap_pairs.groupby(user_col).size().rename('overlap_count')

    score_union = users_a.merge(
        users_b,
        how='outer',
        on=[user_col, similar_user_col],
    )
    score_union['a_score'] = score_union['a_score'].fillna(0.0)
    score_union['b_score'] = score_union['b_score'].fillna(0.0)
    score_union['a_weight'] = score_union['a_score'].clip(lower=0)
    score_union['b_weight'] = score_union['b_score'].clip(lower=0)
    score_union['weighted_min'] = np.minimum(score_union['a_weight'], score_union['b_weight'])
    score_union['weighted_max'] = np.maximum(score_union['a_weight'], score_union['b_weight'])
    score_union['score_product'] = score_union['a_score'] * score_union['b_score']
    score_union['a_score_sq'] = score_union['a_score'] ** 2
    score_union['b_score_sq'] = score_union['b_score'] ** 2

    score_metrics = score_union.groupby(user_col).agg(
        weighted_min_sum=('weighted_min', 'sum'),
        weighted_max_sum=('weighted_max', 'sum'),
        score_dot=('score_product', 'sum'),
        a_score_norm_sq=('a_score_sq', 'sum'),
        b_score_norm_sq=('b_score_sq', 'sum'),
    )
    score_metrics['weighted_jaccard'] = np.divide(
        score_metrics['weighted_min_sum'],
        score_metrics['weighted_max_sum'],
        out=np.zeros(len(score_metrics), dtype=float),
        where=score_metrics['weighted_max_sum'].to_numpy() != 0,
    )
    score_cosine_denominator = np.sqrt(score_metrics['a_score_norm_sq'] * score_metrics['b_score_norm_sq'])
    score_metrics['score_cosine'] = np.divide(
        score_metrics['score_dot'],
        score_cosine_denominator,
        out=np.zeros(len(score_metrics), dtype=float),
        where=score_cosine_denominator.to_numpy() != 0,
    )

    details = (
        pd.DataFrame(index=user_ids)
        .join(a_count)
        .join(b_count)
        .join(overlap_count)
        .join(score_metrics[['weighted_jaccard', 'score_cosine']])
        .fillna(0)
    )
    details[['a_count', 'b_count', 'overlap_count']] = details[['a_count', 'b_count', 'overlap_count']].astype(int)
    details['overlap_pct_a'] = np.divide(
        details['overlap_count'],
        details['a_count'],
        out=np.zeros(len(details), dtype=float),
        where=details['a_count'].to_numpy() != 0,
    )
    details['overlap_pct_b'] = np.divide(
        details['overlap_count'],
        details['b_count'],
        out=np.zeros(len(details), dtype=float),
        where=details['b_count'].to_numpy() != 0,
    )

    union_count = details['a_count'] + details['b_count'] - details['overlap_count']
    details['jaccard'] = np.divide(
        details['overlap_count'],
        union_count,
        out=np.zeros(len(details), dtype=float),
        where=union_count.to_numpy() != 0,
    )
    details = details.reset_index()

    summary = pd.DataFrame([{
        'top_n': top_n,
        'users_a': int(users_a[user_col].nunique()),
        'users_b': int(users_b[user_col].nunique()),
        'users_compared': int(len(details)),
        'users_with_overlap': int((details['overlap_count'] > 0).sum()),
        'mean_overlap_count': float(details['overlap_count'].mean()) if not details.empty else 0.0,
        'mean_overlap_pct_a': float(details['overlap_pct_a'].mean()) if not details.empty else 0.0,
        'mean_overlap_pct_b': float(details['overlap_pct_b'].mean()) if not details.empty else 0.0,
        'mean_jaccard': float(details['jaccard'].mean()) if not details.empty else 0.0,
        'mean_weighted_jaccard': float(details['weighted_jaccard'].mean()) if not details.empty else 0.0,
        'mean_score_cosine': float(details['score_cosine'].mean()) if not details.empty else 0.0,
    }])

    if return_details:
        return summary, details, overlap_pairs
    return summary


###################################  
def calc_half_decay(n0, t, t_half):
    #half decay
    #N(t) = N0 * (1/2)^(t / T1/2)
    nt = n0 * (0.5 ** (t / t_half))
    return nt

###################################
def calc_rating(attendance_df, events = None, today = pd.Timestamp.today().normalize(), recs_params=None):
    attendance_df = attendance_df.copy()
    
    t_half = recs_params.get('time_half_decay',365)
    attendance_type_weights = recs_params.get('attendance_type_weights',{'default': 0,'scheduled': 1,'waitlisted': 4,'attended': 5,'bookmark': 4})
    
    attendance_df['rating'] = attendance_type_weights['default']
    attendance_df.loc[attendance_df['ATTENDANCE_STATUS'].isin(['scheduled']),'rating'] = attendance_type_weights['scheduled']
    attendance_df.loc[attendance_df['ATTENDANCE_STATUS'].isin(['waitlisted']),'rating'] = attendance_type_weights['waitlisted']
    attendance_df.loc[attendance_df['ATTENDANCE_STATUS'].isin(['attended']),'rating'] = attendance_type_weights['attended']
    attendance_df.loc[attendance_df['ATTENDANCE_STATUS'].isin(['bookmark']),'rating'] = attendance_type_weights['bookmark']
    
    if('SESSION_DATETIME' not in attendance_df.columns):        
        logger.info("SESSION_DATETIME does not exist in attendance data. Joining with events.")        
        attendance_df = attendance_df.merge(events[['EVENTCODE','START_DATE']], how = 'left', on = 'EVENTCODE')
    else:
        attendance_df['START_DATE'] = attendance_df['SESSION_DATETIME']
        
    today = pd.Timestamp(today).normalize()
    attendance_df["days"] = (today - pd.to_datetime(attendance_df["START_DATE"])).dt.days    
    new_rating = attendance_df.apply(lambda row: calc_half_decay(row['rating'], row["days"], t_half), axis=1)
    attendance_df.drop(columns=['START_DATE', 'days'], inplace = True)

    return new_rating

###################################
def get_user_history(user_id, attendance, bookmarks, sessions, events):
    sessions_times = get_session_min_time(sessions, events)
    user_attendance = attendance.loc[attendance['ATTENDEE_ID'] == user_id, ['ATTENDEE_ID','EVENTCODE','SESSION_ID','ATTENDEDDATE','ATTENDEDTIME','ATTENDANCE_STATUS']]
    user_attendance['ATTENDANCE_DATETIME'] = pd.to_datetime(user_attendance["ATTENDEDDATE"].astype(str) + " " + user_attendance["ATTENDEDTIME"].astype(str), errors="coerce")
    user_attendance = user_attendance[['ATTENDEE_ID','EVENTCODE','SESSION_ID','ATTENDANCE_DATETIME','ATTENDANCE_STATUS']]
    user_attendance = user_attendance.merge(sessions[['SESSION_ID','EVENTCODE', 'TITLE']].rename(columns={'TITLE':'SESSION_TITLE'}), how = 'left', on =['SESSION_ID','EVENTCODE'])
    user_attendance = user_attendance.merge(sessions_times, how = 'left', on =['SESSION_ID'])
    user_attendance = user_attendance[['ATTENDEE_ID','EVENTCODE','SESSION_ID','SESSION_DATETIME','ATTENDANCE_DATETIME','ATTENDANCE_STATUS','SESSION_TITLE']]
    #user_attendance
    
    user_bookmarks = bookmarks.loc[bookmarks['ATTENDEE_ID'] == user_id, ['ATTENDEE_ID', 'EVENTCODE', 'SESSION_ID', 'SESSION_DATE', 'SESSION_TIME', 'SESSION_TITLE', 'ATTENDED_DATETIME']].rename(columns={'ATTENDED_DATETIME':'ATTENDANCE_DATETIME'})
    user_bookmarks['SESSION_DATETIME'] = pd.to_datetime(user_bookmarks["SESSION_DATE"].astype(str) + " " + user_bookmarks["SESSION_TIME"].astype(str), errors="coerce")
    user_bookmarks[['ATTENDANCE_STATUS']] = 'bookmark'
    user_bookmarks = user_bookmarks[['ATTENDEE_ID', 'EVENTCODE', 'SESSION_ID', 'ATTENDANCE_DATETIME', 'ATTENDANCE_STATUS', 'SESSION_DATETIME','SESSION_TITLE']]
    #user_bookmarks
    
    user_history = pd.concat([user_attendance,user_bookmarks]).sort_values('SESSION_DATETIME', ascending = False)
    #sessions_times['SESSION_ID'].value_counts()
    #len(set(sessions_times['SESSION_ID']))
    
    return user_history
###################################
