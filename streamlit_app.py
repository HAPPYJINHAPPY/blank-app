import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the uploaded file
file_path = 'corrected_fatigue_simulation_data.csv'
data = pd.read_csv(file_path)

# 1. 特征和标签
X = data.drop(columns=["Fatigue_Label"])
y = data["Fatigue_Label"]

# 统一特征列名，避免空格
X.columns = X.columns.str.replace(' ', '_')

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型训练
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict(X_test)

# 5. 评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 训练一个简单的模型
model = RandomForestClassifier()
model.fit(X, y)

# 保存模型
with open("fatigue_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ----------------------------------------
# 1. 加载机器学习模型
# ----------------------------------------
@st.cache_resource
def load_model():
    # 替换为你的模型路径
    with open("fatigue_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# ----------------------------------------
# 2. 构建 Streamlit 应用
# ----------------------------------------
st.title("Fatigue Prediction Dashboard")

# 创建输入表单
st.sidebar.header("Input Parameters")


def user_input_features():
    neck_flexion = st.sidebar.slider("Neck Flexion", 0, 90, 45)
    neck_extension = st.sidebar.slider("Neck Extension", 0, 90, 10)
    shoulder_elevation = st.sidebar.slider("Shoulder Elevation", 0, 180, 90)
    shoulder_forward = st.sidebar.slider("Shoulder Forward", 0, 90, 30)
    elbow_flexion = st.sidebar.slider("Elbow Flexion", 0, 180, 90)
    wrist_extension = st.sidebar.slider("Wrist Extension", 0, 90, 15)
    wrist_deviation = st.sidebar.slider("Wrist Deviation", 0, 45, 10)
    back_flexion = st.sidebar.slider("Back Flexion", 0, 90, 20)
    task_duration = st.sidebar.number_input("Task Duration (seconds)", min_value=0, value=600)
    movement_frequency = st.sidebar.number_input("Movement Frequency (times/min)", min_value=0, value=10)

    # 将输入汇总为一个 DataFrame
    data = {
        "Neck_Flexion": neck_flexion,
        "Neck_Extension": neck_extension,
        "Shoulder_Elevation": shoulder_elevation,
        "Shoulder_Forward": shoulder_forward,
        "Elbow_Flexion": elbow_flexion,
        "Wrist_Extension": wrist_extension,
        "Wrist_Deviation": wrist_deviation,
        "Back_Flexion": back_flexion,
        "Task_Duration": task_duration,
        "Movement_Frequency": movement_frequency,
    }
    return pd.DataFrame(data, index=[0])


input_data = user_input_features()

# 显示用户输入数据
st.subheader("Input Parameters")
st.write(input_data)

# ----------------------------------------
# 3. 预测与结果展示
# ----------------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    st.write(f"Fatigue Level: {prediction[0]}")
