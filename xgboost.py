import pandas as pd
import matplotlib.pyplot as plt

mon_shop_item_cnt = sales_train[
    ['date_block_num','shop_id','item_id','item_cnt_day'] # item price or datefull
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'item_cnt_day':'mon_shop_item_cnt'})
for_graph1=mon_shop_item_cnt[['date_block_num','mon_shop_item_cnt']].groupby('date_block_num').sum()
mon_shop_item_sales = sales_train[
    ['date_block_num','shop_id','item_id','item_price']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'item_price':'mon_shop_item_sales'})
for_graph2=mon_shop_item_sales[['date_block_num','mon_shop_item_sales']].groupby('date_block_num').sum()

sns.heatmap(for_graph2)
train_df=mon_shop_item_cnt.merge(mon_shop_item_sales)
item_categories['big_category_name'] = item_categories['item_category_name'].map(lambda x: x.split(' - ')[0])
item_categories.loc[
    item_categories['big_category_name']=='Чистые носители (штучные)','big_category_name'
] = 'Чистые носители'


item_categories.loc[
    item_categories['big_category_name']=='Чистые носители (шпиль)','big_category_name'
] = 'Чистые носители'

item_categories['big_category_name'].value_counts()
train_df=mon_shop_item_cnt.merge(mon_shop_item_sales)

for_graph3= train_df.groupby(
    ['date_block_num','big_category_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=for_graph3,hue='big_category_name')
plt.title('Montly item counts by big category')

train_full_comb = pd.DataFrame()
for i in range(35):
    mid = test[['shop_id','item_id']]
    mid['date_block_num'] = i
    train_full_comb = pd.concat([train_full_comb,mid],axis=0)

train = pd.merge(
    train_full_comb,
    mon_shop_item_cnt,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)

train = pd.merge(
    train,
    items[['item_id','big_category_name']],
    on='item_id',
    how='left'
)

lag_col_list = ['mon_shop_item_cnt','mon_shop_item_sales']
lag_num_list = [1,2,3,6,9,12,]

train = train.sort_values(
    ['shop_id', 'item_id','date_block_num'],
    ascending=[True, True,True]
).reset_index(drop=True)

for lag_col in lag_col_list:
    for lag in lag_num_list:
        set_col_name =  lag_col + '_' +  str(lag)
        df_lag = train[['shop_id', 'item_id','date_block_num',lag_col]].sort_values(
            ['shop_id', 'item_id','date_block_num'],
            ascending=[True, True,True]
        ).reset_index(drop=True).shift(lag).rename(columns={lag_col: set_col_name})
        train = pd.concat([train, df_lag[set_col_name]], axis=1)

from sklearn.preprocessing import LabelEncoder
obj_col_list = ['big_category_name','city_name']
for obj_col in obj_col_list:
    le = LabelEncoder()
    train_[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train_[obj_col])})
    test_[obj_col] = pd.DataFrame({obj_col:le.fit_transform(test_[obj_col])})

train_y = train_['mon_shop_item_cnt']
train_X = train_.drop(columns=['mon_shop_item_cnt','mon_shop_item_sales','date_block_num'])
test_X = test_.drop(columns=['mon_shop_item_cnt','mon_shop_item_sales','date_block_num'])
import xgboost as xgb
dm_train = xgb.DMatrix(train_X, label=train_y)
param = {
    'max_depth': 10, 
    'eta': 1, 
    'objective': 'reg:squarederror'
}
model = xgb.train(param, dm_train)

xgb.plot_importance(model)
xgb.to_graphviz(model)
dm_test = xgb.DMatrix(test_X)
y_pred = model.predict(dm_test)
y_pred
test_y = model.predict(dm_test)
test_X['item_cnt_month'] = test_y
submission = pd.merge(
    test,
    test_X[['shop_id','item_id','item_cnt_month']],
    on=['shop_id','item_id'],
    how='left'
)
submission[['ID','item_cnt_month']].to_csv('submission.csv', index=False)