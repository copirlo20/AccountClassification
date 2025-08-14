import pandas as pd

cols = ['id', 'dataset']
users = pd.concat([
    pd.read_csv("./cresci-2015/E13/users.csv"),
    pd.read_csv("./cresci-2015/FSF/users.csv"),
    pd.read_csv("./cresci-2015/INT/users.csv"), 
    pd.read_csv("./cresci-2015/TFP/users.csv"),
    pd.read_csv("./cresci-2015/TWT/users.csv"),
], ignore_index=True)[cols]

users_id = set(users['id'])

friends = pd.concat([
    pd.read_csv("./cresci-2015/E13/friends.csv"),
    pd.read_csv("./cresci-2015/FSF/friends.csv"),
    pd.read_csv("./cresci-2015/INT/friends.csv"), 
    pd.read_csv("./cresci-2015/TFP/friends.csv"),
    pd.read_csv("./cresci-2015/TWT/friends.csv"),
], ignore_index=True)

edges = pd.concat([
    friends[['source_id', 'target_id']], 
    friends[['target_id', 'source_id']].rename(columns={'target_id': 'source_id', 'source_id': 'target_id'})
])
edges = edges.drop_duplicates().query("source_id != target_id").reset_index(drop=True)
edges = edges.merge(users, left_on='source_id', right_on='id').rename(columns={'dataset': 'source_dataset'}).drop(columns='id')
edges = edges.merge(users, left_on='target_id', right_on='id').rename(columns={'dataset': 'target_dataset'}).drop(columns='id')

edges.to_csv("./cresci-2015/edges.csv", index=False)