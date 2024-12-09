import streamlit as st
import pandas as pd
import numpy as np
import pickle


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
        "Neck Flexion": neck_flexion,
        "Neck Extension": neck_extension,
        "Shoulder Elevation": shoulder_elevation,
        "Shoulder Forward": shoulder_forward,
        "Elbow Flexion": elbow_flexion,
        "Wrist Extension": wrist_extension,
        "Wrist Deviation": wrist_deviation,
        "Back Flexion": back_flexion,
        "Task Duration": task_duration,
        "Movement Frequency": movement_frequency,
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

