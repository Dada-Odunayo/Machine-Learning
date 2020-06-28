import pandas as pd
df = pd.read_csv("salaries.csv")
inputs = df.drop('salary_more_than_100k',axis='columns')
target = df['salary_more_than_100k']
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_new'] = le_company.fit_transform(inputs['company'])
inputs['job_new'] = le_company.fit_transform(inputs['job'])
inputs['degree_new'] = le_company.fit_transform(inputs['degree'])

print(inputs.head)

inputs_new = inputs.drop(['company','job','degree'], axis = 'columns')

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_new, target)
u=model.score(inputs_new,target)
print(u*100,'%')
e=model.predict([[2,2,1]])
print(e)