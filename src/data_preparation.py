import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


#  DATASET 1: TITANIC

def load_titanic(path='Titanic-Dataset.csv'):
    df = pd.read_csv(path)

    # 1) delete irrelevant columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # 2) fill missing values
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 3) Encoding from text to numeric
    df['Sex']      = LabelEncoder().fit_transform(df['Sex'])       # female=0, male=1
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])  # C=0, Q=1, S=2

    # 4)split features and target
    X = df.drop(columns=['Survived']).values
    y = df['Survived'].values

    # 5) Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 6) Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[Titanic] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Titanic] Classes: {np.unique(y)} → 0=No survived, 1=Survived\n")
    return X_train, X_test, y_train, y_test

#  DATASET 2: DIABETES

def load_diabetes(path='diabetes.csv'):
    df = pd.read_csv(path)

    # Split feautures and target
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[Diabetes] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Diabetes] Classes: {np.unique(y)} → 0=Healthy, 1=Diabetes\n")
    return X_train, X_test, y_train, y_test


#  MAIN 
if __name__ == '__main__':
    print("=" * 50)
    print("  PHASE 1 — DATA PREPARATION")
    print("=" * 50)
    print()

    X_train_t, X_test_t, y_train_t, y_test_t = load_titanic(r'C:\Users\calda\DatMining\Data-Mining-Analysis\datasets\Titanic-Dataset.csv')
    X_train_d, X_test_d, y_train_d, y_test_d = load_diabetes(r'C:\Users\calda\DatMining\Data-Mining-Analysis\datasets\diabetes.csv')

    print(" Both datasets ready for BPN.")