
import pandas as pd
import numpy as np

def blight_model():
    train=pd.read_csv('train.csv', encoding='ISO-8859-1')
    test=pd.read_csv('test.csv')
    
    
    train=train[~train['compliance'].isnull()]

    list_to_remove_all = ['fine_amount', 'violator_name', 'zip_code', 'country', 'city',
                                      'inspector_name', 'violation_street_number', 'violation_street_name',
                                      'violation_zip_code', 'violation_description',
                                      'mailing_address_str_number', 'mailing_address_str_name',
                                      'non_us_str_code',
                                     'grafitti_status', 'violation_code']
    train.drop(list_to_remove_all,1,inplace=True)
    test.drop(list_to_remove_all,1,inplace=True)

    list_to_remove_train = [
                    'balance_due',
                    'collection_status',
                    'compliance_detail',
                    'payment_amount',
                    'payment_date',
                    'payment_status']

    train.drop(list_to_remove_train,1,inplace=True)


    add =  pd.read_csv('addresses.csv')
    latlon = pd.read_csv('latlons.csv')

    new=pd.merge(add,latlon,how='left',left_on='address',right_on='address')
    new.drop('address',1,inplace=True)
    train_join=pd.merge(train,new,how='left',left_on='ticket_id',right_on='ticket_id')
    test_join=pd.merge(test,new,how='left',left_on='ticket_id',right_on='ticket_id')

    train_join=train_join.set_index('ticket_id')
    test_join=test_join.set_index('ticket_id')
    train=train_join
    test=test_join

    to_split=['disposition','state','agency_name']

    train.lat.fillna(method='pad', inplace=True)
    train.lon.fillna(method='pad', inplace=True)
    train.state.fillna(method='pad', inplace=True)

    test.lat.fillna(method='pad', inplace=True)
    test.lon.fillna(method='pad', inplace=True)
    test.state.fillna(method='pad', inplace=True)

    train=pd.get_dummies(train,columns=to_split)
    test=pd.get_dummies(test,columns=to_split)

    from datetime import datetime

    def diff(hear,issue):
                if not hear or type(hear)!=str: return 73
                return ( datetime.strptime(hear , "%Y-%m-%d %H:%M:%S") - datetime.strptime(issue , "%Y-%m-%d %H:%M:%S")).days
    train=train[~train['hearing_date'].isnull()]
    train['delay'] = train.apply(lambda data: diff(data['hearing_date'], data['ticket_issued_date']), axis=1)
    test['delay'] = test.apply(lambda data: diff(data['hearing_date'], data['ticket_issued_date']), axis=1)

    train.drop(labels=['ticket_issued_date','hearing_date'],axis=1,inplace=True)
    test.drop(labels=['ticket_issued_date','hearing_date'],axis=1,inplace=True)

    a,b=set(train.columns),set(test.columns)
    for feature in train.columns:
        if(feature not in b):
            a.remove(feature)
    features_taken=list(a)


    X_train=train[features_taken]
    X_test=test[features_taken]
    y_train=train.compliance
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler

    scaler=MinMaxScaler()
    X_fit_train= scaler.fit_transform(X_train)
    X_fit_test=scaler.transform(X_test)

    nn=MLPClassifier(random_state=0,hidden_layer_sizes=[100,50],alpha=5,solver='lbfgs')
    nn.fit(X_fit_train,y_train)

    compliance=nn.predict_proba(X_fit_test)[:,1]
    compliance=pd.Series(compliance,index=test.index)
    return compliance      
        
    




