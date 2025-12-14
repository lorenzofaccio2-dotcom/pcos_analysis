# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE  # serve per classi non bilanciate
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
warnings.filterwarnings('ignore')

data1 = pd.read_csv(r"C:\Users\loren\Downloads\archive\PCOS_infertility.csv")
data2 = pd.read_excel(
r"C:\Users\loren\Downloads\archive\PCOS_data_without_infertility.xlsx", sheet_name='Full_new')



data1.select_dtypes(include=np.number).hist(figsize=(10, 8))
plt.show()

#Merge the files
data = pd.merge(data2, data1, on='Patient File No.', suffixes=('', '_wo'), how='left')


data =data.drop(['Sl. No_wo','II    beta-HCG(mIU/mL)_wo', 'Unnamed: 44','PCOS (Y/N)_wo', '  I   beta-HCG(mIU/mL)_wo'
       , 'AMH(ng/mL)_wo'], axis=1)

data = data.rename(columns = {"PCOS (Y/N)":"Dependent variable"})

data = data.drop(["Sl. No","Patient File No."],axis = 1)

# null values
valori_nulli = data.isnull().sum()
print(valori_nulli)


def Variabili_numeriche(df):
    numerical_cols = df.select_dtypes(include='number').columns
    stats = {}
    for col in numerical_cols:
        stats[col] = {
            "media": df[col].mean(),
            "mediana": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    stats = pd.DataFrame(stats)
    return stats
print(Variabili_numeriche(data))
stats_df = Variabili_numeriche(data).T  # Trasposto per tabella più leggibile
colors = ['#5B2A86', '#2C6975', '#CA2E55', '#2F2F2F', '#EFD9CE']


plots = {}
numerical_cols = data.select_dtypes(include='number').columns

for col in numerical_cols:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.countplot(x=col, data=data, ax=ax)
    plt.title(col)
    plt.xticks(rotation=55)

    plots[col] = fig

    plt.close(fig)

#first_key = list(plots.keys())[5]
#plots[first_key].show()

for col in data.select_dtypes(include='object').columns:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))


# Imoutare valori Mancanti
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


# boxplots fper variabili numeriche
plots_box = {}

for col in numerical_cols:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(x=col, data=data, ax=ax, palette=colors)
    plt.title(col)
    plt.xticks(rotation=55)
    plots_box[col] = fig
    plt.close(fig)

# key_box = list(plots_box.keys())[5]
#plots_box[key_box].show()





# analisi secifiche
def correlation(df):
    correlation = data.select_dtypes(include='number').corr()
    mask = (correlation > 0.4) | (correlation < -0.4)
    correlation = correlation.where(mask)
    correlation = correlation.stack().reset_index()
    correlation.columns = ["Var1", "Var2", "Correlation"]
    correlation = correlation[correlation["Var1"] != correlation["Var2"]]
    return correlation

print(correlation(data))

# studio della matrice di correlazione e PCA iniziale
corr = data.select_dtypes(include='number').corr()
eigenvalues, eigenvectors = np.linalg.eig(corr)

index = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[index]
eigenvectors = eigenvectors[:, index]

print("Autovalori:")
print(eigenvalues)

print("\nAutovettori:")
print(eigenvectors)

explained_variance = eigenvalues / np.sum(eigenvalues)
plt.figure(figsize=(10,5))
plt.plot(range(1, len(eigenvalues)+1), explained_variance, marker='o')
plt.title("Varianza spiegata da ciascuna componente")
plt.xlabel("Componente")
plt.ylabel("Varianza spiegata")
plt.grid(True)
plt.show()

# derivata discreta per elbow method
delta = np.diff(explained_variance)
print(delta)

# 3/4 componente principale spiega tanta varianza

# distribution of the dependent variable

sns.countplot(data = data, x = 'Dependent variable')
plt.title("Dependent variable Distribution")
plt.show()    # attenzione, classi sbilanciate

X = data.drop(columns=['Dependent variable'])  # Features
y = data['Dependent variable']  # Target

# apply the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# always apply SMOTE after the split (othgerwise data leakage)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

sns.countplot(x = y_train)
plt.title("Dependent variable Distribution after Smote")
plt.show()    # adesso abbiamo classi bilanciate

# inserire modelli in dizionario per ciclare sopra
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}


def evaluate_model(name, y_test, y_pred, y_score=None):
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    if y_score is not None:
        print("AUC:", roc_auc_score(y_test, y_score))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# e qui ciclo sui modelli inizializzati in models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    evaluate_model(name, y_test, y_pred, y_score)

stack_models = [
    ('rf', models['Random Forest']),
    ('nb', models['Naive Bayes']),
    ('xgb', models['XGBoost']),
    ('ada', models['AdaBoost'])
]

results = []

for cv in [3,5,7,10]:
    meta = RandomForestClassifier(random_state=42)
    model = StackingClassifier(
        estimators=stack_models,
        final_estimator=meta,
        cv=cv
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]

    results.append({
        "cv_internal": cv,
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_score) # qui nell'auc si usa la classe di probabilità
    })

print(pd.DataFrame(results)) # usare il 5 cv




