import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as sds, LabelEncoder

train = pd.read_csv('synth-ecom-data-2025\synthetic_ecommerce_sales_2025.csv')

print("Path to dataset files:", train)

# train.info() # ringkasan info dataset
train.describe(include='all') # statistik deskriptif

# memeriksa nan
miss_val = train.isnull().sum()
miss_val[miss_val > 0]

# memisahkan kolom
less = miss_val[miss_val < 0].index
over = miss_val[miss_val >= 0].index

# menisi nan dengan median
num_features = train[less].select_dtypes(include=['number']).columns
train[num_features] = train[num_features].fillna(train[num_features].median())

# mengisi nan dengan modus
cat_features = train[less].select_dtypes(include=['object']).columns
for col in cat_features:
    train[col] = train[col].fillna(train[col].mode()[0])

df = train.drop(columns=over) # hapus kolom dengan banyak nan

# verifikasi
miss_val = df.isnull().sum()
miss_val[miss_val > 0]

# outliers handling
for feature in num_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

# identifikasi outlier dengan iqr
Q1 = df[num_features].quantile(0.25)
Q3 = df[num_features].quantile(0.75)
IQR = Q3 - Q1

# hapus outlier
cond = ~((df[num_features] < (Q1 - 1.5 * IQR)) | (df[num_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_flt_num = df.loc[cond, num_features]

# tidak menghapus outlier
'''
med = df[num_features].median()
df[num_features] = df[num_features].apply(lambda x: x.median if x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR) else x)
df[num_features] = df[num_features].apply(lambda x: (Q1 - 1.5 * IQR) if x < Q1 else (Q3 + 1.5 * IQR) if x > Q3 else x)
''' # mengisi dengan nilai terdekat

# rejoin dengan kolom kategori
cat_features = df.select_dtypes(include=['object']).columns
df = pd.concat([df_flt_num, df.loc[cond, cat_features]], axis=1)

# standarisasi
scaler = sds()
df[num_features] = scaler.fit_transform(df[num_features])

# histogram before
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train[num_features[3]], kde=True)
plt.title('Before Preprocessing')

# histogram after
plt.subplot(1, 2, 2)
sns.histplot(df[num_features[3]], kde=True)
plt.title('After Preprocessing')

# identifikasi duplikat
duplicates = df.duplicated()
print(f'Duplicate rows: {duplicates}')
df = df.drop_duplicates() # hapus duplikat
print(f'After removing duplicates: {df}')

# one hot encoding
df_one = pd.get_dummies(df, columns=cat_features)
df_one.info()

# inisialisasi label encoder
lbl_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in cat_features:
    df_lencoder[col] = lbl_encoder.fit_transform(df[col])
df_lencoder.info() # hasil