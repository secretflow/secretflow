import pandas as pd

origin_data_path = (
    "/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/data/criteo_train_small.csv"
)
origin_df = pd.read_csv(origin_data_path)

sample_data = origin_df.sample(frac=0.01)
columns = sample_data.columns

train_alice = sample_data.sample(frac=0.5, axis='columns')
train_bob = sample_data.drop(columns=train_alice.columns)

if 'label' not in train_alice.columns:
    train_alice['label'] = sample_data['label']
    train_bob = train_bob.drop(columns=['label'])


train_alice.to_csv(
    "/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/dcn_split/train_alice.csv",
    index=False,
    sep="|",
    encoding='utf-8',
)
train_bob.to_csv(
    "/root/develop/Open_Source/ant-sf/secretflow/oscp/DCN/dcn_split/train_bob.csv",
    index=False,
    sep="|",
    encoding='utf-8',
)
