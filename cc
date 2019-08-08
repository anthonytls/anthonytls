

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt



# Exploration et Préparation des données

Etudiez chaque attribut et ses caractéristiques :
— Nom de la variable
— Type (catégorique, int / float, texte, structuré, etc.)
— Pourcentage de valeurs manquantes
— Bruit et type de bruit (stochastique, valeurs aberrantes, erreurs d’arrondi, etc.)
— Peut-être utile pour la tâche ?
— Type de distribution (gaussienne, uniforme, logarithmique, etc.)

_ les corrélations entre les variables


Nettoyage des données
— Corriger ou supprimer les valeurs aberrantes
— Complétez les valeurs manquantes (par exemple, avec zéro, moyenne, médiane. . . )
ou supprimez leurs lignes

Feature selection : Supprimez les attributs qui ne fournissent aucune information
utile pour la tâche

Feature engineering
— Discrétiser les variables continues
— Décomposer les caractéristiques (catégorielles, date / heure, etc.).
— Ajoutez des transformations(e.g., log(x), sqrt(x), x2, etc.)
— Regroupez certaine variables
— Normalisation

# Apprentissage et Validation des modèles

#


# 


"""
数据基本处理
"""
"""
数据基本处理
"""

# 常用包 --------------------------------------------------------
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
np.random.seed(42)


# 加载数据 --------------------------------------------------------
df = pd.read_csv("oecd_bli_2015.csv", thousands=',')
df = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1', na_values="n/a")
df_code = pd.read_csv('Association Alias Exp PTF PEC 20190405.csv', thousands=',',encoding = 'unicode_escape') 
pd.read_excel('tmp.xlsx', index_col=0) 
df.read_csv('temp.csv', dtype={'your_column': str})

# 数据本地序列化操作  --------------------------------------------------------
df.to_csv('../gen/df.csv', columns=df.columns, index=True)


# 批量从文件中合并csv --------------------------------------------------------
import glob
path = r'/Users/anthony/Desktop/test' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    li.append(df)
frame = pd.concat(li, axis=1, ignore_index=True)



# 基本分析 --------------------------------------------------------

# Base的基本信息
df.head()
df.tail(5)
df.shape
df.columns
df.info() # 方法可以快速查看数据的描述，特别是总行数、每个属性的类型和非空值的数量
df.describe()
 数据格式的问题

df.dtypes
df.Weight.astype('int64') 


# 类型型变量

df['$a'].value_counts().sort_values(ascending=False)  # 基本统计

df["var"].value_counts() # 类别变量统计个数

df.team.astype('category')
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin') 


cat_cols = data.select_dtypes(['category']).columns # 选出类别型变量
data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
data.head()


cabin_tab = pd.crosstab(index=titanic_train["Cabin"],  # Make a crosstab
                        columns="count")  

检验类别型变量之间的关系
策略1 抽样都比较一下
策略2 根据认知

dummies = pd.get_dummies(df.categorical) # 二元化 然后删除老变量
df.join(dummies)
df.drop([var],axis=1)


# 数值型变量




df.corr() # 连续变量的相关系数







# 数据变换操作 --------------------------------------------------------

df.index
df.columns # 重新定义列项名称
df.values
df.loc[行标签,列标签]
df.loc['2014-07-23':'2014-07-31', 'open']   #字段型找

df.iloc[:, 10] # 选择第11列  #数值型找
df.iloc[1:5, 2:6]

df[['列标签']]   # 根据列名称选特定列 返回还是dataframe

# 指定一个列
df.close[0:3]
df[['close', 'high', 'low']][0:3]
df[df.columns[0]]


# 逻辑条件进行数据筛选 --------------------------------------------------------

#对Var进行赛选 var2 更换值  根据一列产生新的一列
df['positive'] = np.where(df.p_change > 0, 1, 0)


df[np.abs(df.p_change) > 8]
df[(np.abs(df.p_change) > 8) & (df.volume > 2.5 * df.volume.mean())]
# 两个条件 注意在条件里加括号!!!!


df[df['user_id']==30]  # 逻辑判断

df.ix[df['Var']== "V1",'Var2'] = 1

#dropping rows from dataframe based on a “not in” condition

a = ['2015-01-01' , '2015-02-01']

df = pd.DataFrame(data={'date':['2015-01-01' , '2015-02-01', '2015-03-01' , '2015-04-01', '2015-05-01' , '2015-06-01']})

print(df)
#         date
#0  2015-01-01
#1  2015-02-01
#2  2015-03-01
#3  2015-04-01
#4  2015-05-01
#5  2015-06-01

df = df[~df['date'].isin(a)]
For "IN" use: something.isin(somewhere)

Or for "NOT IN": ~something.isin(somewhere)





# Drop Duplicate Rows --------------------------------------------------------
df = df.drop_duplicates()   # fully duplicated
df = df.sort_values('Age', ascending=False)
df = df.drop_duplicates(subset='Name', keep='first') # partially duplicated and keep the first 


# drop rows with condition 
# Get names of indexes for which column Age has value 30
indexNames = dfObj[ dfObj['Age'] == 30 ].index
# Delete these row indexes from dataFrame
dfObj.drop(indexNames , inplace=True)
data_evt.iloc[index]


# 数据转换与缺失值规整 --------------------------------------------------------

# 缺失值
import missingno as msno
msno.matrix(df, labels=True)

df.isnull().mean().sort_values(ascending=False)* 100 # 每个变量缺失值占比

thresh = len(df) * .2
df.dropna(thresh = thresh, axis = 1, inplace = True)

# counting the number of missing/NaN in each row
df.isnull().sum(axis=1)



# 排序
df.sort_values(by='p_change')
df.sort_values(by='p_change', ascending=False)

# 如果一行的数据中存在na就删除这行
tsla_df.dropna()            
# 通过how控制 如果一行的数据中全部都是na就删除这行
tsla_df.dropna(how='all') 
df.drop(columns=['B', 'C'])
# pandas drop columns using list of column names
gapminder_ocean.drop(['pop', 'gdpPercap', 'continent'], axis=1)

# 使用指定值填充na， inplace代表就地操作，即不返回新的序列在原始序列上修改
tsla_df.fillna(tsla_df.mean(), inplace=True).head()


# 构建交叉表 --------------------------------------------

xt = pd.crosstab(tsla_df.date_week, tsla_df.positive)
xt_pct = xt.div(xt.sum(1).astype(float), axis=0) # 换成比列

xt_pct.plot(
    figsize=(8, 5),
    kind='bar',
    stacked=True,
    title='date_week -> positive')
plt.xlabel('date_week')
plt.ylabel('positive')

# 构建透视表 --------------------------------------------
df.pivot_table(['positive'], index=['date_week'])

Base.pivot_table(index='Pretime', columns='Target1', values='INST_REF',aggfunc='count')
df.groupby(['date_week', 'positive'])['positive'].count()
df_.groupby(['INST_REF'])['EVT_CD'].transform(lambda x: accumulate(x))   # 分组累加object 自定义函数
df['no_csum'] = df.groupby(['name'])['no'].cumsum()  # 分组累加

print(df.groupby('A').head())   


g = df.groupby('A')
g['B'].tolist()

# convert groupeby objectt to dataframe
df_ = pd.DataFrame(data_evt.groupby(['INST_REF'])['CREA_INST_DH'].max()).reset_index()


# concat, append, merge的使用 ------------------------------
# 合并两个df
pandas.concat([df1[:], df2[:]], axis=1)	合并两个df
pd.merge(stock_a, stock_b, left_on='stock_a', right_on='stock_b')
res =  res1.merge(res2, on='Pretime', how='left').merge(res3, on='Pretime', how='left')
df.drop_duplicates() #注意检查




# 给dataframe的表格上色 ---------------------------------------
http://pandas.pydata.org/pandas-docs/stable/user_guide/style.html



# 快速作图 ------------------------------------------------------------------------------
pandas.Series.plot	
pandas.Dataframe.biboxplot	

# 直接把数值型变量画出来 直接感受 所有！
df.plot()  
df.hist(bins=50, figsize=(20,15))
df.var.hist(bins=80)
plt.show()

#同时对比画两个变量
df[['close', 'volume']].plot(subplots=True, style=['r', 'g'], grid=True)


#  变量转化 ---------------------------------------

# 对行或则列加一个函数
df['salary'] = df['SAL-RATE'].apply(money_to_float)


# 直接把所有的变量变成类型变量
df = df.astype(object)

#  convert the categorical columns with string values to numeric representations



技巧 ------------------------------------------------------------------------

isinstance() # 就可以告诉我们，一个对象是否是某种类型
dir() # 如果要获得一个对象的所有属性和方法，可以使用函数


# 时间处理 ------------------------------------------------------------------------
# 计算时间
%%time

import time
import datetime
 
#str 转datetime
str = '2012-11-19'
date_time = datetime.datetime.strptime(str,'%Y-%m-%d')
print date_time
 
#获取当前时间
t_date=datetime.date.today()
print t_date
#获取第几周
week = t_date.strftime("%W")
print(week)
#获取星期几
week_day = t_date.strftime("%w")


pd.to_datetime(var, format = '%d%m%Y %H:%M:%S', errors ='ignore')

# only the date in return from the timestamp

date_time.date()

# 随机抽样
df.sample(n)

# Create a list of date from range
import datetime
base = datetime.datetime.today().date()
date_list = [base - datetime.timedelta(days=x) for x in range(0, 4)]




DataFrame.apply operates on entire rows or columns at a time.
DataFrame.applymap, Series.apply, and Series.map operate on one element at time.





set(list1) | set(list2)	# union	# 包含 list1 和 list2 所有数据的新集合
set(list1) & set(list2)	# intersection	包含 list1 和 list2 中共同元素的新集合
set(list1) - set(list2)	# difference	在 list1 中出现但不在 list2 中出现的元素的集合



Scikit-learn balanced subsampling
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys



# Train and Test Datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
X_train.shape, X_test.shape



# 回归问题 ------------------------------------------------------------------------

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor


lm = linear_model.LinearRegression()
dt = DecisionTreeRegressor(random_state = 0)
rr = Ridge(alpha=0.01)
rr10 = Ridge(alpha=100)
lasso = Lasso()
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
svr = SVR(gamma='scale', C=1.0, epsilon=0.2)
mlp = MLPRegressor(hidden_layer_sizes=(3,),
                                     activation='relu',
                                     solver='adam',
                                     learning_rate='adaptive',
                                     max_iter=1000,
                                     learning_rate_init=0.01,
                                     alpha=0.01)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
adb = AdaBoostRegressor(random_state=0, n_estimators=100)


models = [lm,dt,rr,rr10,lasso,
          #lasso001,lasso00001,svr,
          rf,mlp,adb,gb]

for regressor in models:
    model = regressor.fit(X=X_train, y=y_train)
    y_hat_train = regressor.predict(X_train)
    y_hat_test = regressor.predict(X_test)
    print_score(y_hat_train,y_hat_test, regressor)


def print_score(y_hat_train,y_hat_test, reg):
    print("Regressor: ", reg.__class__.__name__)
    print('Train MAPE: ', round(((abs(y_train - y_hat_train))).mean(), 3))
    print('Test MAPE: ', round(((abs(y_test - y_hat_test))).mean(), 3))
    print('Score Train: ', reg.score(X_train,y_train))
    print('Score Test: ', reg.score(X_test, y_test))
    print("---")  




# Features importances for RF
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',  ascending=False)


# 分类问题 ------------------------------------------------------------------------

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score



lr = LogisticRegression()
# mn = MultinomialNB()
Bayes = BernoulliNB()
boosting = GradientBoostingClassifier(learning_rate=0.1,random_state=10)
knn = KNeighborsClassifier(n_neighbors=5) 
rf= RandomForestClassifier(n_estimators=100, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0)
# lsvm =LinearSVC(multi_class='crammer_singer')
#xgb = XGBClassifier(max_depth=5, n_estimators=100, colsample_bytree=0.8, 
    #                    subsample=0.8, nthread=10, learning_rate=0.1)
models = [lr,  Bayes, boosting, knn, rf]


for classifier in models:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(y_pred, classifier)
    







# 模型选择问题 ------------------------------------------------------------------------

# Gridsearch 
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, ShuffleSplit, train_test_split 
def GradientBooster(param_grid, n_jobs): 
    estimator = GradientBoostingRegressor() 
    #Choose cross-validation generator - let's choose ShuffleSplit which randomly shuffles and selects Train and CV sets 
    #for each iteration. There are other methods like the KFold split. 
    cv = ShuffleSplit(X_train.shape[0], test_size=0.2)
    
    #Apply the cross-validation iterator on the Training set using GridSearchCV.  
    
    classifier = RandomizedSearchCV(estimator=estimator, cv=cv, param_distributions=param_grid, n_jobs=n_jobs)
    
    #We'll now fit the training dataset to this classifier 
    classifier.fit(X_train, y_train) 
    #Let's look at the best estimator that was found by GridSearchCV 
    
    print ( "Best Estimator learned through GridSearch" )
    print (classifier.best_estimator_ )

    return cv, classifier.best_estimator_ 


param_grid={'n_estimators':[100], 
            'learning_rate': [0.1,0.05], #, 0.02, 0.01],
            'max_depth':[4,6], 
            'min_samples_leaf':[3,5],#,9,17], 
            'max_features':[1.0,0.3]#,0.1] 
           } 

n_jobs= 4 #Let's fit GBRT to the digits training dataset by calling the function we just created. 
            
cv,best_est=GradientBooster(param_grid, n_jobs) 

print ("Best Estimator Parameters")
print("---------------------------")
print ("n_estimators: %d" %best_est.n_estimators)
print ("max_depth: %d" %best_est.max_depth)
print ("Learning Rate: %.1f" %best_est.learning_rate)
print ("min_samples_leaf: %d" %best_est.min_samples_leaf)
print ("max_features: %.1f" %best_est.max_features)
print ("Train R-squared: %.2f" %best_est.score(X_train,y_train))



# Croise validation 
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
time_split = TimeSeriesSplit(n_splits=5)
[(el[0].shape, el[1].shape) for el in time_split.split(X_train)] 

def cv_mae(model, X=X_train ):
    mae = -cross_val_score(model, X, y_train, scoring='neg_mean_absolute_error', cv=time_split, n_jobs=1)
    return (mae)

def cv_f1_macro(model, X=X_train):
    f1 = cross_val_score(OneVsRestClassifier(model), X, y_train, scoring='f1_macro', cv=time_split, n_jobs=1)
    return (f1)






# 画图查看结果
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



from sklearn.learning_curve import learning_curve 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)): 
    """ Generate a simple plot of the test and traning learning curve. 
    Parameters ---------- 
    estimator : object type that implements the "fit" and "predict" methods 
        An object of that type which is cloned for each validation. 
    
    title : string 
        Title for the chart.
    
    X : array-like, shape (n_samples, n_features) 
        Training vector, where n_samples is the number of samples and n_features is the number of features. 
    
    y : array-like, shape (n_samples) or (n_samples, n_features), optional 
        Target relative to X for classification or regression; 
    None for unsupervised learning. 
    
    ylim : tuple, shape (ymin, ymax), optional 
        Defines minimum and maximum yvalues plotted. 
    
    cv : integer, cross-validation generator, optional 
        If an integer is passed, it is the number of folds (defaults to 3). 
        Specific cross-validation objects can be passed, see sklearn.cross_validation module for the list of possible objects 
    
    n_jobs : integer, optional 
        Number of jobs to run in parallel (default 1).
        
         """ 
    plt.figure() 
    plt.title(title) 
    if ylim is not None: 
        plt.ylim(*ylim) 
    plt.xlabel("Training examples") 
    plt.ylabel("Score") 
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1) 
    train_scores_std = np.std(train_scores, axis=1) 
    test_scores_mean = np.mean(test_scores, axis=1) 
    test_scores_std = np.std(test_scores, axis=1) 
    plt.grid() 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r") 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score") 
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score") 
    plt.legend(loc="best") 
    return( plt)




# Plot for confusion matrix - for classification ---------------------------------------
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, 
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred,  normalize=True,
                      title='Normalized confusion matrix')

plt.show()
