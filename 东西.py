from volcenginesdkarkruntime import Ark
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
from matplotlib import font_manager
import os

font_path = "C:/Users/X2006936/Downloads/SourceHanSansCN-Normal.otf"  # 替换为你的上传字体文件名

# 检查字体文件是否存在
if not os.path.exists(font_path):
    st.error(f"Font file not found: {font_path}")
else:
    # 设置字体属性
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()

    # 创建自定义函数来统一设置字体
    def set_font_properties(ax, font_prop):
        """统一设置坐标轴和标题字体"""
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
        ax.title.set_fontproperties(font_prop)
        ax.xaxis.label.set_fontproperties(font_prop)
        ax.yaxis.label.set_fontproperties(font_prop)
    # 全局设置字体
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

# Load the uploaded file
file_path = 'corrected_fatigue_simulation_data.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 1. Features and labels
X = data.drop(columns=["疲劳等级"])
y = data["疲劳等级"]

# Normalize column names to avoid spaces
X.columns = X.columns.str.replace(' ', '_')

# 2. Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save model
with open("fatigue_model.pkl", "wb") as f:
    pickle.dump(model, f)

@st.cache_resource
def load_model():
    with open("fatigue_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# 创建 Ark 客户端
API_KEY = st.text_input("请输入 OpenAI API 密钥", type="password")
if not API_KEY:
    st.info("请输入 OpenAI API 密钥以继续。", icon="🗝️")
else:
    client = Ark(api_key=API_KEY)

    # 显示标题和介绍
    st.title("疲劳评估与 AI 聊天助手")
    st.write(
        "这个应用结合了疲劳评估模型和 AI 聊天机器人，提供基于用户输入的疲劳评估结果和实时建议。"
    )

    # 初始化存储所有预测记录的列表
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []

    # 输入疲劳评估参数
    col1, col2 = st.columns(2)

    with col1:
        neck_flexion = st.slider("颈部前屈", 0, 60, 20)
        neck_extension = st.slider("颈部后仰", 0, 60, 25)
        shoulder_elevation = st.slider("肩部上举范围", 0, 180, 60)
        shoulder_forward = st.slider("肩部前伸范围", 0, 90, 30)

    with col2:
        elbow_flexion = st.slider("肘部屈伸", 0, 180, 90)
        wrist_extension = st.slider("手腕背伸", 0, 90, 15)
        wrist_deviation = st.slider("手腕桡偏/尺偏", 0, 45, 10)
        back_flexion = st.slider("背部屈曲范围", 0, 90, 20)

    # 输入任务参数
    st.subheader("持续时间")
    task_duration = st.number_input("持续时间（秒）", min_value=0, value=6)
    movement_frequency = st.number_input("重复频率（每分钟）", min_value=0, value=5)

    # 汇总输入数据
    input_data = pd.DataFrame({
        "颈部前屈": [neck_flexion],
        "颈部后仰": [neck_extension],
        "肩部上举范围": [shoulder_elevation],
        "肩部前伸范围": [shoulder_forward],
        "肘部屈伸": [elbow_flexion],
        "手腕背伸": [wrist_extension],
        "手腕桡偏/尺偏": [wrist_deviation],
        "背部屈曲范围": [back_flexion],
        "持续时间": [task_duration],
        "重复频率": [movement_frequency],
    })

    st.subheader("输入参数")
    st.write(input_data)

    # 评估按钮
    result = None  # 确保变量 result 初始化
    if st.button("评估"):
        with st.spinner("正在评估，请稍等..."):
            # 模型预测
            prediction = model.predict(input_data)
            result = ["低疲劳状态", "中疲劳状态", "高疲劳状态"][prediction[0]]
            st.success(f"评估结果：{result}")

            # 保存评估记录
            record = input_data.copy()
            record["评估"] = result
            if 'predictions' not in st.session_state:
                st.session_state.predictions = []
            st.session_state.predictions.append(record)

    # 右侧空白区域扩展为 AI 分析助手
    if result is not None:
        # 创建一个额外的右侧布局
        with st.container():
            st.subheader("AI 智能评估助手")
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": "你是一个疲劳评估助手，基于用户的疲劳状态和角度数据提供建议。"}]

            # AI 输入构造
            ai_input = f"用户的疲劳状态是：{result}。\n" \
                       f"用户提供的角度数据为：颈部前屈{neck_flexion}度，颈部后仰{neck_extension}度，" \
                       f"肩部上举范围{shoulder_elevation}度，肩部前伸范围{shoulder_forward}度，" \
                       f"肘部屈伸{elbow_flexion}度，手腕背伸{wrist_extension}度，" \
                       f"手腕桡偏/尺偏{wrist_deviation}度，背部屈曲范围{back_flexion}度。\n" \
                       f"请基于这些数据给出用户的潜在人因危害分析及改善建议。"

            st.session_state.messages.append({"role": "user", "content": ai_input})

            # 显示现有聊天记录
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])


                def call_ark_api(messages):
                    try:
                        ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
                        completion = client.chat.completions.create(
                            model="ep-20241226165134-6lpqj",  # 使用正确的 Ark 模型ID
                            messages=ark_messages,
                            stream=True
                        )

                        response = ""
                        for chunk in completion:
                            delta_content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta,
                                                                                      "content") else ""
                            yield delta_content
                    except Exception as e:
                        st.error(f"调用 Ark API 时出错：{e}")
                        yield f"Error: {e}"


                # 创建占位符显示机器人回答
                response_placeholder = st.empty()
                response = ""
                for partial_response in call_ark_api(st.session_state.messages):
                    response += partial_response
                    response_placeholder.markdown(response)

                # 将 AI 回复保存到会话状态
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("请先点击评估按钮生成结果后再查看分析。")
