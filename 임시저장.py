# %%
import pandas as pd
import numpy as np

codeA = pd.read_csv('data/Train/tc_codea_코드A.csv')
codeB = pd.read_csv('data/Train/tc_codeb_코드B.csv')
code_area = pd.read_csv('data/Train/tc_sgg_시군구코드.csv')

# 라벨링 데이터

# 그 뭘 했었는지 - 방문장소 , 여기서 방문한 장소의 타입을 찾을 수 있음
visit_area_info = pd.read_csv('data/Train/tn_visit_area_info_방문지정보_merged.csv')
traveler_master = pd.read_csv('data/Train/tn_traveller_master_여행객 Master_merged.csv')

# 돈은 얼마나 쓰는지 - 지출내역
adv_consume_his_act = pd.read_csv('data/Train/tn_activity_consume_his_활동소비내역_t.csv')

# 뭘 타고는지 - 이동수단소비내역
move_consume_his = pd.read_csv('data/Train/tn_mvmn_consume_his_이동수단소비내역_merged.csv')

companion_info = pd.read_csv('data/Train/tn_companioninfo동반자정보_t.csv')

# 여행 목적 - 여행 페르소나
travel = pd.read_csv('./data/Train/tn_travel_여행_merged.csv')

# TODO: 감정 (무드)를 추출해보기 위한 데이터 추후 추가 필요요

# %%
valid_area_type_codes = list(range(1, 9))
visit_area_info = visit_area_info[ (visit_area_info['VISIT_AREA_TYPE_CD'].isin(valid_area_type_codes))]
print(f"데이터 크기: {len(visit_area_info)}")

# %%
visit_area_info = visit_area_info.reset_index(drop = True)

# %%
visit_area_info['VISIT_AREA_TYPE_CD'].unique()

# %%
visit_area_info.dropna(subset = ['LOTNO_ADDR'], inplace = True)
visit_area_info = visit_area_info.reset_index(drop = True)

# %%
sido = []
gungu = []
for i in range(len(visit_area_info['LOTNO_ADDR'])):
    sido.append(visit_area_info['LOTNO_ADDR'][i].split(' ')[0])
    gungu.append(visit_area_info['LOTNO_ADDR'][i].split(' ')[1])

# %%
visit_area_info['SIDO'] = sido
visit_area_info['GUNGU'] = gungu

# %%
visit_area_info['SIDO'].value_counts()

# %%
visit_area_info = visit_area_info[['TRAVEL_ID', 'VISIT_AREA_NM', 'SIDO', 'GUNGU', 'VISIT_AREA_TYPE_CD', 'DGSTFN',
                                'REVISIT_INTENTION', 'RCMDTN_INTENTION', 'RESIDENCE_TIME_MIN', 'REVISIT_YN']]

# %%
# TRAVEL_MISSION_CHECK의 첫번째 항목 가져오기
travel_list = []
for i in range(len(travel)):
    value = int(travel['TRAVEL_MISSION_CHECK'][i].split(';')[0])
    travel_list.append(value)

travel['TRAVEL_MISSION_PRIORITY'] = travel_list

# %%
travel = travel[['TRAVEL_ID', 'TRAVELER_ID', 'TRAVEL_MISSION_PRIORITY']]

# %%
traveler_master = traveler_master[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'INCOME', 'TRAVEL_STYL_1', 
                                     'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 
                                     'TRAVEL_STYL_6', 'TRAVEL_STYL_7','TRAVEL_STYL_8', 
                                      'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM' ]]

# %%
valid_codes = list(range(1, 8))  # 1부터 7까지의 리스트
adv_consume_his_act = adv_consume_his_act[
    (adv_consume_his_act['ACTIVITY_TYPE_CD'].isin(valid_codes))
]
print(f"결측치 제거 전 데이터 크기: {len(adv_consume_his_act)}")

# %%
# 결측치 제거
adv_consume_his_act = adv_consume_his_act.dropna(subset=['STORE_NM'])
adv_consume_his_act = adv_consume_his_act[['TRAVEL_ID', 'ACTIVITY_TYPE_CD', 'STORE_NM']]

print(f"결측치 제거 후 데이터 크기: {len(adv_consume_his_act)}")

# %% [markdown]
# # 동반자 정보
#   - 1182,TCR,"1",배우자,"","",N,999,"2022-09-22 17:34:50",
#   - 1183,TCR,"2",자녀,"","",N,999,"2022-09-22 17:34:50",
#   - 1184,TCR,"3",부모,"","",N,999,"2022-09-22 17:34:50",
#   - 1185,TCR,"4",조부모,"","",N,999,"2022-09-22 17:34:50",
#   - 1186,TCR,"5",형제/자매,"","",N,999,"2022-09-22 17:34:50",
#   - 1187,TCR,"6",친인척,"","",N,999,"2022-09-22 17:34:50",
#   - 1188,TCR,"7",친구,"","",N,999,"2022-09-22 17:34:50",
#   - 1189,TCR,"8",연인,"","",N,999,"2022-09-22 17:34:50",
#   - 1190,TCR,"9",동료,"","",N,999,"2022-09-22 17:34:50",
#   - 1191,TCR,"10","친목 단체/모임(동호회, 종교단체 등)","","",N,999,"2022-09-22 17:34:50",
#   - 1192,TCR,"11",기타,"","",N,999,"2022-09-22 17:34:50"

# %%
valid_companion_codes = list(range(1, 11)) # 1부터 10까지의 리스트
companion_info = companion_info[(companion_info['REL_CD'].isin(valid_companion_codes))]

companion_info = companion_info[['TRAVEL_ID', 'REL_CD']]

# %%
move_consume_his = move_consume_his[['TRAVEL_ID', 'MVMN_SE_NM']]

# %%
df = pd.merge(travel, traveler_master, left_on = 'TRAVELER_ID', right_on = 'TRAVELER_ID', how = 'inner')

# %%
df = pd.merge(visit_area_info, df, left_on = 'TRAVEL_ID', right_on = 'TRAVEL_ID', how = 'right')

# %%
df = pd.merge(companion_info, df, left_on = 'TRAVEL_ID', right_on = 'TRAVEL_ID', how = 'right')

# %%
df = pd.merge(adv_consume_his_act, df, left_on = 'TRAVEL_ID', right_on = 'TRAVEL_ID', how = 'right')

# %%
df = pd.merge(move_consume_his, df, left_on = 'TRAVEL_ID', right_on = 'TRAVEL_ID', how = 'right')

# %%
df['RESIDENCE_TIME_MIN'] = df['RESIDENCE_TIME_MIN'].replace(0,60)

# %%
df['REVISIT_YN'] = df['REVISIT_YN'].replace("N",0)
df['REVISIT_YN'] = df['REVISIT_YN'].replace("Y",1)

# %%
df.dropna(subset = ['TRAVEL_STYL_1'], inplace = True)
df.reset_index(drop= True, inplace = True)

# %%
df.dropna(subset = ['TRAVEL_MOTIVE_1'], inplace = True)
df.reset_index(drop= True, inplace = True)

# %%
df.shape

# %%
print(df)
df.to_csv('svd_results.csv', index=False)

# %%
df.isna().sum().sum()


