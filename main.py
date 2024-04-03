from data_preprocessing import load_and_preprocess_data
from model import train_model

data_path = "/content/drive/MyDrive/lewagon_project/5sec"
X, y = load_and_preprocess_data(data_path)

X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

model, history = train_model(X_train, y_train, X_val, y_val, X_test, y_test)
