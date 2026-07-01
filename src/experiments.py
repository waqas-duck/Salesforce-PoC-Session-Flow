#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:03:11 2026

@author: mdraminski
"""

import missing_utils as util

attendee_hit_ratio_df = attendee_hit_ratio
attendee_hit_ratio_df = attendee_hit_ratio_df.rename(columns = {'attendees_pct': str(len(event_similarities.columns))})

ret_recs_size = 30
sim_size = 1
past_attendance_real = past_attendance.loc[~past_attendance['SESSION_ID'].isin(event_sessions['SESSION_ID'])].copy()
sim_size_list = [len(event_similarities.columns), 100, 50, 25, 10, 5, 3, 2, 1]
for sim_size in sim_size_list:
    event_similarities = sessions_sims[event_sessions['SESSION_ID']].copy()
    event_similarities = event_similarities.loc[~event_similarities.index.isin(event_sessions['SESSION_ID'])]
    event_similarities = sutil.update_similarities(event_similarities, top_n = sim_size, mult = 0.01)        
    ret_recs = sutil.calc_user_recommendations(event_users_before, past_attendance_real, event_similarities, ret_recs_size=ret_recs_size)        
    attendee_hit_ratio, attendee_hit_cnt = sutil.calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
    
    attendee_hit_ratio_df[str(sim_size)] = attendee_hit_ratio['attendees_pct']

attendee_hit_ratio_df

sim_size
# top_n_recs     358     100      50      25      10       5       3       2       1
# 0           3  0.4441  0.3657  0.3604  0.3750  0.4392  0.4871  0.5371  0.5581  0.6033
# 1           5  0.5764  0.4839  0.4795  0.5383  0.6060  0.6335  0.7178  0.6960  0.7339
# 2          10  0.7463  0.6550  0.6763  0.7397  0.7854  0.8330  0.8608  0.8640  0.8855
# 3          15  0.8191  0.7490  0.7759  0.8311  0.8679  0.8984  0.9092  0.9148  0.9275
# 4          20  0.8755  0.8176  0.8367  0.8818  0.9097  0.9294  0.9358  0.9385  0.9460
# 5          25  0.9043  0.8530  0.8755  0.9116  0.9336  0.9448  0.9512  0.9541  0.9656
# 6          30  0.9255  0.8845  0.9058  0.9351  0.9458  0.9541  0.9590  0.9683  0.9717



today = pd.Timestamp.today().normalize()
today = pd.Timestamp(event_date).normalize()

attendee_hit_ratio_df = attendee_hit_ratio
attendee_hit_ratio_df = attendee_hit_ratio_df.rename(columns = {'attendees_pct': str(len(event_similarities.columns))})

ret_recs_size = 30
sim_size = 1
t_half_list = [720,420,360,180,60,30]
for t_half in t_half_list:
    past_attendance_real = past_attendance.loc[~past_attendance['SESSION_ID'].isin(event_sessions['SESSION_ID'])].copy()
    rating = sutil.calc_rating(past_attendance, events, today = today, t_half = t_half)
    past_attendance_real['rating'] = rating    
    event_similarities = sessions_sims[event_sessions['SESSION_ID']].copy()
    event_similarities = event_similarities.loc[~event_similarities.index.isin(event_sessions['SESSION_ID'])]
    event_similarities = sutil.update_similarities(event_similarities, top_n = sim_size, mult = 0.01)    
    ret_recs = sutil.calc_user_recommendations(event_users_before, past_attendance_real, event_similarities, ret_recs_size=ret_recs_size)        
    attendee_hit_ratio, attendee_hit_cnt = sutil.calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])    
    attendee_hit_ratio_df[str(t_half)] = attendee_hit_ratio['attendees_pct']

attendee_hit_ratio_df


# top_n_recs  attendees_pct     720     420     360     180      60      30
# 0           3         0.6033  0.6067  0.6064  0.6067  0.5928  0.5593  0.5388
# 1           5         0.7339  0.7324  0.7297  0.7283  0.7271  0.7085  0.6917
# 2          10         0.8855  0.8899  0.8877  0.8870  0.8826  0.8752  0.8411
# 3          15         0.9275  0.9299  0.9304  0.9307  0.9333  0.9285  0.8992
# 4          20         0.9460  0.9470  0.9478  0.9485  0.9497  0.9451  0.9204
# 5          25         0.9656  0.9666  0.9670  0.9670  0.9683  0.9641  0.9407
# 6          30         0.9717  0.9724  0.9724  0.9727  0.9736  0.9705  0.9487



attendee_hit_ratio_df = attendee_hit_ratio
attendee_hit_ratio_df = attendee_hit_ratio_df.rename(columns = {'attendees_pct': str(5)})

cold_start_users = list(set(user_similarities['ATTENDEE_IN']))
len(cold_start_users)
ret_recs_size = 30
user_nn_size = 5
user_nn_size_list = [25,20,15,10,5,3,2,1]

for user_nn_size in user_nn_size_list:    
    ret_recs = sutil.calc_cold_start_recommendations(cold_start_users, user_similarities, past_attendance, event_similarities, user_nn_size, ret_recs_size)    
    attendee_hit_ratio, attendee_hit_cnt = sutil.calc_recs_stats(ret_recs, event_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])    
    attendee_hit_ratio_df[str(user_nn_size)] = attendee_hit_ratio['attendees_pct']


attendee_hit_ratio_df









#select event_session_similarity
event_session_similarity = session_similarity[event_sessions['SESSION_ID']].copy()
event_session_similarity = event_session_similarity.loc[~event_session_similarity.index.isin(event_sessions['SESSION_ID'])]
event_session_similarity = sutil.update_similarities(event_session_similarity, top_n = session_sim_size, mult = 0.01)
logger.info("event_session_similarity size: %s", event_session_similarity.shape)
#event_session_similarity = None


##############################
### GET past_attendance AND event_attendance
event_attendance = attendance.loc[(attendance['EVENTCODE'].isin([eventcode])) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))),]
logger.info(util.show_object_memory(event_attendance, "event_attendance"))                                   
past_attendance = attendance.loc[(attendance['EVENTCODE'].isin(past_events)) & (attendance['SESSION_ID'].isin(set(sessions['SESSION_ID']))) & (attendance['ATTENDEE_ID'].isin(event_users_returning_id)),]
logger.info(util.show_object_memory(past_attendance, "past_attendance"))

import random
sample_users_id = random.sample(event_users_returning_id,3000)
past_attendance = past_attendance.loc[past_attendance['ATTENDEE_ID'].isin(sample_users_id),]
event_attendance = event_attendance.loc[event_attendance['ATTENDEE_ID'].isin(sample_users_id),]


import importlib
import optutils as outil
outil = importlib.reload(outil)
print(outil.optimize_attendance_type_weights_genetic.__code__.co_filename)

opt_result_gen4 = outil.optimize_attendance_type_weights_genetic(
    sample_users_id,
    past_attendance,
    event_attendance,
    events,
    event_session_similarity,
    event_date,
    ret_recs_size=30,
    t_half=365,
    user_sessions_filter=None,
    recs_top_n=15,
    recs_top_n_list=None,
    weight_min=1,
    weight_max=5,
    weight_values=None,
    weight_keys=('scheduled', 'waitlisted', 'attended', 'bookmark'),
    fixed_weights=None,
    population_size=10,
    epochs=10,
    elite_size=2,
    mutation_rate=0.2,
    tournament_size=3,
    random_state=None,    
    default_attendance_type_weights = {'default': 0,'scheduled': 1,'waitlisted': 4,'attended': 5,'bookmark': 4},
    use_default_event_attendance_weights=True,
    progress_every=1)


opt_result_gen4['results'].to_csv('~/opt.res.gen4.csv')


sample_users_id = random.sample(event_users_returning_id,5000)

ret_recs_popular = sutil.get_popular_sessions(session_room, ret_recs_size, None).copy()
ret_recs_popular = sutil.get_recs_popular_for_users(sample_users_id, ret_recs_popular)
len(set(ret_recs_popular['ATTENDEE_ID']))

ret_recs_regular = ret_recs.loc[ret_recs['ATTENDEE_ID'].isin(sample_users_id)]
len(set(ret_recs_regular['ATTENDEE_ID']))
    
event_session_embeddings = session_embeddings.select(list(event_sessions["SESSION_ID"]))
ret_recs_nn = sutil.calc_cold_start_recs_nn(sample_users_id, past_attendance, attendee_similarity, event_session_similarity, user_sim_size, ret_recs_size).copy()
len(set(ret_recs_nn['ATTENDEE_ID']))

ret_recs_emb = sutil.calc_cold_start_recs_emb(sample_users_id, attendee_embeddings, event_session_embeddings, ret_recs_size).copy()
len(set(ret_recs_emb['ATTENDEE_ID']))


###### CALC RET_RECS STATS
event_sample_attendance = event_attendance.loc[event_attendance['ATTENDEE_ID'].isin(sample_users_id)]

attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = sutil.calc_recs_stats(ret_recs_regular, event_sample_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
ndcg_regular = sutil.calc_ndcg(ret_recs_regular,event_sample_attendance , recs_top_n=15, return_details = True)
ndcg_regular[0]


attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = sutil.calc_recs_stats(ret_recs_popular, event_sample_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
ndcg_popular = sutil.calc_ndcg(ret_recs_popular,event_sample_attendance , recs_top_n=15, return_details = True)
ndcg_popular[0]


attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = sutil.calc_recs_stats(ret_recs_popular, event_sample_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
ndcg_nn = sutil.calc_ndcg(ret_recs_nn,event_sample_attendance , recs_top_n=15, return_details = True)
ndcg_nn[0]


attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = sutil.calc_recs_stats(ret_recs_popular, event_sample_attendance, recs_top_n = 15, recs_top_n_list = [3,5,10,15,20,25,30])
ndcg_emb = sutil.calc_ndcg(ret_recs_emb,event_sample_attendance , recs_top_n=15, return_details = True)
ndcg_emb[0]
