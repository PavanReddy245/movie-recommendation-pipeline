from pathlib import Path
import pandas as pd


def _feature_tables(history, users, items, reference_time=None):
    history=history.copy()
    history['timestamp']=pd.to_datetime(history.timestamp,utc=True)
    reference_time=pd.Timestamp(reference_time) if reference_time is not None else history.timestamp.max()
    user=history.groupby('user_id').agg(user_avg_rating=('rating','mean'),user_activity_count=('movie_id','count'),user_last_interaction=('timestamp','max')).reset_index()
    user['user_days_since_last_interaction']=(reference_time-user.user_last_interaction).dt.total_seconds()/86400
    preferred=history.merge(items[['movie_id','genre']],on='movie_id',how='left').groupby('user_id').genre.agg(lambda s:s.mode().iloc[0] if not s.mode().empty else 'unknown').rename('user_preferred_genre').reset_index()
    user=user.merge(preferred,on='user_id',how='left')
    item=history.groupby('movie_id').agg(item_popularity=('user_id','count'),item_avg_rating=('rating','mean')).reset_index().merge(items,on='movie_id',how='right')
    item['item_popularity']=item.item_popularity.fillna(0).astype(int)
    interaction=history[['user_id','movie_id','rating','timestamp']].copy()
    interaction['days_since_reference']=(reference_time-interaction.timestamp).dt.total_seconds()/86400
    return user,item,interaction,reference_time


def build_features(train, users, items, output_dir: str):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    user,item,interaction,reference_time=_feature_tables(train,users,items)
    user=user.merge(users,on='user_id',how='left')
    for name,df in [('user_features.csv',user),('item_features.csv',item),('interaction_features.csv',interaction)]: df.to_csv(out/name,index=False)
    catalog=pd.DataFrame([
      ['user_avg_rating','user','Mean rating given by user','mean(rating)','float'],
      ['user_activity_count','user','Number of historical interactions','count(movie_id)','integer'],
      ['user_last_interaction','user','Latest historical interaction','max(timestamp)','datetime'],
      ['user_days_since_last_interaction','user','Days since latest historical interaction','reference_time - user_last_interaction','float'],
      ['user_preferred_genre','user','Most frequently interacted genre','mode(genre)','string'],
      ['age','user','User age from metadata','users.age','integer'],['region','user','User region from metadata','users.region','string'],
      ['item_popularity','item','Historical interaction count','count(user_id)','integer'],
      ['item_avg_rating','item','Mean historical rating','mean(rating)','float'],
      ['genre','item','Movie genre from metadata','items.genre','string'],['release_year','item','Movie release year','items.release_year','integer'],
      ['days_since_reference','interaction','Age of event relative to reference timestamp','reference_time - timestamp','float']
    ],columns=['feature_name','entity','description','calculation','data_type'])
    catalog.to_csv(out/'feature_catalog.csv',index=False)
    (out/'feature_reference.json').write_text(__import__('json').dumps({'reference_time':reference_time.isoformat()},indent=2))
    return user,item,interaction,catalog


def build_inference_features(request_users, request_items, historical_interactions, users, items, reference_time=None):
    """Generate prediction-time features using the same definitions as training.

    request_users/request_items are ID collections. Historical interactions must
    contain only events available at prediction time to prevent leakage.
    """
    history=historical_interactions.copy()
    user,item,interaction,ref=_feature_tables(history,users,items,reference_time)
    user=user.merge(users,on='user_id',how='left')
    user=user[user.user_id.isin(list(request_users))].copy()
    item=item[item.movie_id.isin(list(request_items))].copy()
    return user,item,ref
