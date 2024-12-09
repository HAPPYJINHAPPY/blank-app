import streamlit as st
import requests
import matplotlib.pyplot as plt
st.title("🎈Fatigue Prediction Dashboard")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# 分组布局
col1, col2, col3 = st.columns(3)
with col1:
    neck_flexion = st.number_input('Neck Flexion (°)', min_value=0, max_value=90)
    neck_extension = st.number_input('Neck Extension (°)', min_value=0, max_value=90)
with col2:
    shoulder_elevation = st.number_input('Shoulder Elevation (°)', min_value=0, max_value=180)
    shoulder_forward = st.number_input('Shoulder Forward (°)', min_value=0, max_value=180)
with col3:
    elbow_flexion = st.number_input('Elbow Flexion (°)', min_value=0, max_value=180)
    wrist_extension = st.number_input('Wrist Extension (°)', min_value=0, max_value=90)

# 下方布局
wrist_deviation = st.number_input('Wrist Deviation (°)', min_value=0, max_value=90)
back_flexion = st.number_input('Back Flexion (°)', min_value=0, max_value=90)
task_duration = st.number_input('Task Duration (seconds)', min_value=0.0)
movement_frequency = st.number_input('Movement Frequency (Hz)', min_value=0.0)

# 提交预测请求
if st.button('Predict'):
    # 构造输入数据
    input_data = {
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

    try:
        # 发送 POST 请求到后端
        response = requests.post("http://http://172.30.20.64:5000/predict", json=input_data)
        prediction = response.json().get('prediction')
        st.success(f"Predicted Fatigue Label: {prediction}")

        # 可视化输入特征分布
        st.subheader("Feature Values Visualization")
        feature_names = [
            'Neck Flexion', 'Neck Extension', 'Shoulder Elevation',
            'Shoulder Forward', 'Elbow Flexion', 'Wrist Extension',
            'Wrist Deviation', 'Back Flexion', 'Task Duration', 'Movement Frequency'
        ]
        feature_values = [
            neck_flexion, neck_extension, shoulder_elevation, shoulder_forward,
            elbow_flexion, wrist_extension, wrist_deviation, back_flexion,
            task_duration, movement_frequency
        ]

        # 绘制柱状图
        fig, ax = plt.subplots()
        ax.barh(feature_names, feature_values, color='skyblue')
        ax.set_xlabel('Feature Values')
        ax.set_title('Input Features Distribution')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
