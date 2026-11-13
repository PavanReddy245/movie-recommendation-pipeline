import numpy as np
import pandas as pd


def _dcg(relevances):
    return float(sum((2**int(rel)-1)/np.log2(idx+2) for idx,rel in enumerate(relevances)))


def evaluate_ranking(model,train,test,all_item_ids,items,k=5,threshold=4.0,personalized=False):
    relevant=test[test.rating>=threshold].groupby('user_id').movie_id.apply(set).to_dict()
    seen=train.groupby('user_id').movie_id.apply(set).to_dict()
    users=sorted(set(test.user_id)); p=[]; r=[]; n=[]; d=[]; recommended=set(); genre=items.set_index('movie_id').genre.to_dict()
    evaluated=0
    for user in users:
        actual=relevant.get(user,set())
        recs=model.recommend(user,seen.get(user,set()),all_item_ids,k=k)
        recommended.update(recs)
        if not actual: continue
        evaluated+=1; hits=[int(m in actual) for m in recs]
        p.append(sum(hits)/k); r.append(sum(hits)/len(actual))
        ideal=[1]*min(len(actual),k)+[0]*max(0,k-len(actual))
        denom=_dcg(ideal); n.append(_dcg(hits)/denom if denom else 0.0)
        genres=[genre.get(m,'unknown') for m in recs]
        pairs=[(genres[i],genres[j]) for i in range(len(genres)) for j in range(i+1,len(genres))]
        d.append(sum(a!=b for a,b in pairs)/len(pairs) if pairs else 0.0)
    catalog=set(all_item_ids)
    return {'precision_at_k':float(np.mean(p)) if p else 0.0,'recall_at_k':float(np.mean(r)) if r else 0.0,'ndcg_at_k':float(np.mean(n)) if n else 0.0,'coverage':float(len(recommended)/len(catalog)) if catalog else 0.0,'diversity':float(np.mean(d)) if d else 0.0,'evaluated_users':int(evaluated),'recommendation_users':int(len(users))}


def evaluate_rating(model,test):
    pairs=[(r.user_id,r.movie_id,float(r.rating)) for r in test.itertuples() if r.user_id in model.user_index and r.movie_id in model.item_index]
    if not pairs:return {'rmse':None,'mae':None,'rating_pairs':0}
    actual=np.array([p[2] for p in pairs]); pred=np.array([model.predict(p[0],p[1]) for p in pairs])
    return {'rmse':float(np.sqrt(np.mean((actual-pred)**2))),'mae':float(np.mean(np.abs(actual-pred))),'rating_pairs':len(pairs)}


def compare_models(popularity,svd,train,test,items,output_path,config):
    all_items=sorted(items.movie_id.tolist()); k=config['k']
    pop=evaluate_ranking(popularity,train,test,all_items,items,k,config['relevance_threshold'])
    personalized=evaluate_ranking(svd,train,test,all_items,items,k,config['relevance_threshold'])
    rating=evaluate_rating(svd,test); rows=[]
    for name,metrics in [('Popularity',pop),('Personalized SVD',personalized)]:
        row={'model':name,**metrics}; row.update(rating if name=='Personalized SVD' else {'rmse':None,'mae':None,'rating_pairs':0}); rows.append(row)
    result=pd.DataFrame(rows); result.to_csv(output_path,index=False); return result
