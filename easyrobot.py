import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import streamlit as st
from matplotlib import font_manager
import os
from volcenginesdkarkruntime import Ark
import base64
import requests
import datetime
import io

# 获取文件内容，指定编码为utf-8，避免UnicodeDecodeError
def get_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""  # 如果文件不存在，返回空字符串
    except UnicodeDecodeError:
        st.error("文件编码错误，无法解码文件。")
        return None

# 获取文件的 SHA 值
def get_file_sha(file_path):
    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        return file_info['sha']  # 返回SHA值
    else:
        st.warning(f"无法从 GitHub 获取文件: {response.json()}")
        return None

# 保存数据到 CSV 文件
def save_to_csv(input_data, result):
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = {
        "颈部前屈": int(input_data["颈部前屈"].values[0]),
        "颈部后仰": int(input_data["颈部后仰"].values[0]),
        "肩部上举范围": int(input_data["肩部上举范围"].values[0]),
        "肩部前伸范围": int(input_data["肩部前伸范围"].values[0]),
        "肘部屈伸": int(input_data["肘部屈伸"].values[0]),
        "手腕背伸": int(input_data["手腕背伸"].values[0]),
        "手腕桡偏/尺偏": int(input_data["手腕桡偏/尺偏"].values[0]),
        "背部屈曲范围": int(input_data["背部屈曲范围"].values[0]),
        "持续时间": int(input_data["持续时间"].values[0]),
        "重复频率": int(input_data["重复频率"].values[0]),
        "fatigue_result": result,
        "timestamp": timestamp  # 增加时间戳
    }
    df = pd.DataFrame([data])

    # 检查文件是否存在
    if os.path.exists(FILE_PATH):
        existing_content = get_file_content(FILE_PATH)
        
        # 如果文件内容非空，读取数据
        if existing_content and existing_content.strip():
            existing_df = pd.read_csv(io.StringIO(existing_content))
        else:
            # 如果文件为空，初始化空的 DataFrame
            existing_df = pd.DataFrame(columns=['timestamp', '颈部前屈', '颈部后仰', '肩部上举范围', '肩部前伸范围', '肘部屈伸', '手腕背伸', '手腕桡偏/尺偏', '背部屈曲范围', '持续时间', '重复频率', 'fatigue_result'])
    else:
        # 文件不存在，初始化空的 DataFrame
        existing_df = pd.DataFrame(columns=['timestamp', '颈部前屈', '颈部后仰', '肩部上举范围', '肩部前伸范围', '肘部屈伸', '手腕背伸', '手腕桡偏/尺偏', '背部屈曲范围', '持续时间', '重复频率', 'fatigue_result'])

    # 合并现有的 DataFrame 和新数据
    updated_df = pd.concat([existing_df, df], ignore_index=True)

    # 保存更新后的 DataFrame 到 CSV 文件
    updated_df.to_csv(FILE_PATH, index=False)

# 上传到 GitHub
def upload_to_github(file_path):
    # 获取文件的 SHA 值
    sha_value = get_file_sha(file_path)
    
    # 读取 CSV 文件内容并进行 base64 编码
    with open(file_path, 'rb') as file:
        content = base64.b64encode(file.read()).decode()

    # GitHub API 请求 URL
    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'

    # 提交的信息
    commit_message = "Add new fatigue data with timestamp"
    
    data = {
        "message": commit_message,
        "branch": GITHUB_BRANCH,
        "content": content,
    }
    
    # 如果文件已经存在，提供 sha 值
    if sha_value:
        data["sha"] = sha_value

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    response = requests.put(url, json=data, headers=headers)

    # 输出详细错误信息
    if response.status_code != 200 and response.status_code != 201:
        st.error(f"Failed to upload CSV file to GitHub: {response.json()}")
        print(f"GitHub API Response: {response.json()}")
    else:
        st.success("CSV file successfully uploaded to GitHub!")
        
font_path = "SourceHanSansCN-Normal.otf"  # 替换为你的上传字体文件名

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
file_path = 'corrected_fatigue_simulation_data_Chinese.csv'
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

# Feature importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Create feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
ax.set_title("Feature Importance in Fatigue Classification")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
set_font_properties(ax, font_prop)

# Save model
with open("fatigue_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit sidebar
if st.sidebar.checkbox("Show Model Performance"):
    st.subheader("Model Performance Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    st.write("Classification Report:")
    st.text(report)
    st.pyplot(fig)


@st.cache_resource
def load_model():
    with open("fatigue_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()
# Streamlit sidebar
if st.sidebar.checkbox("标准参考"):
    st.markdown("""各动作的舒适范围：  
    颈部前屈舒适范围：0°到40°  
    颈部后仰舒适范围：0°到40°    
    肩部上举舒适范围：0°到120°（过度上举可能导致肩部肌肉疲劳和肩袖损伤风险）  
    肩部前伸舒适范围：0°到90°  
    肘部屈伸舒适范围：0°到150°   
    手腕背伸舒适范围：0°到60°  
    手腕桡偏 / 尺偏舒适范围：0°到15°（任何超过此角度都容易造成腕管综合症或肌腱问题）  
    背部屈曲舒适范围：0°到45°   
    重复频率（次/分钟）:   
    0 - 20 低疲劳，低风险，适合大多数日常活动  
    20 - 40 中疲劳，中等风险，长时间重复可能导致肌肉疲劳  
    > 40 高疲劳，高风险，极容易导致肌肉劳损、骨骼疲劳和长期伤害  
    参考文献和来源：  
    ISO 11228-3:2003: 适用于评估高重复频率和大范围的手臂、肩膀和颈部动作对工作者的影响。  
    NIOSH: 美国国家职业安全健康研究所关于工作任务和重复动作的疲劳评估模型。  
    OSHA: 美国职业安全健康管理局的工作条件和疲劳评估准则。""")

# 使用 Markdown 居中标题
st.markdown("<h1 style='text-align: center;'>疲劳评估测试系统</h1>", unsafe_allow_html=True)
st.markdown("""模型参考ISO 11226: Static working postures、NIOSH Guidelines for Manual Material Handling、
OWAS Analysis and Recommendations.等国际标准""")

# 初始化存储所有预测记录的列表
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
st.subheader("角度参数")
# Two-column layout for sliders
col1, col2 = st.columns(2)

with col1:
    neck_flexion = st.slider("颈部前屈", 0, 60, 20)
    neck_extension = st.slider("颈部后仰", 0, 60, 25)
    shoulder_elevation = st.slider("肩部上举范围", 0, 180, 60)
    shoulder_forward = st.slider("肩部前伸范围", 0, 180, 120)

with col2:
    elbow_flexion = st.slider("肘部屈伸", 0, 180, 120)
    wrist_extension = st.slider("手腕背伸", 0, 60, 15)
    wrist_deviation = st.slider("手腕桡偏/尺偏", 0, 30, 10)
    back_flexion = st.slider("背部屈曲范围", 0, 60, 20)

# Task parameters
st.subheader("时间参数")
col3, col4 = st.columns(2)
with col3:
    task_duration = st.number_input("持续时间（秒）", min_value=0, value=5)
with col4:
    movement_frequency = st.number_input("重复频率（每5分钟）", min_value=0, value=35)

# 初始化会话状态
if "show_ai_analysis" not in st.session_state:
    st.session_state.show_ai_analysis = False
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'client' not in st.session_state:
    st.session_state.client = None

    # 定义疲劳评估函数
def fatigue_prediction(input_data):
    prediction = model.predict(input_data)
    return ["低疲劳状态", "中疲劳状态", "高疲劳状态"][prediction[0]]


# 定义聊天调用函数
def call_ark_api(client, messages):
    try:
        ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        completion = client.chat.completions.create(
            model="ep-20241226165134-6lpqj",
            messages=ark_messages,
            stream=True
        )
        response = ""
        for chunk in completion:
            delta_content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, "content") else ""
            yield delta_content
    except Exception as e:
        st.error(f"调用 Ark API 时出错：{e}")
        yield f"Error: {e}"


# 输入数据表格
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
st.subheader("参数信息")
st.write(input_data)

# 疲劳评估按钮
if st.button("评估"):
    with st.spinner("正在评估，请稍等..."):
        # 假设 input_data 已经被定义并包含所有必要的数据
        # 请确保 fatigue_prediction 函数已定义
        result = fatigue_prediction(input_data)
        st.success(f"评估结果：{result}")
        timestamp = timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 保存数据到本地 CSV 文件
        save_to_csv(input_data, result)
        upload_to_github(FILE_PATH)
        # 保存评估结果到会话状态
        st.session_state.result = result
        record = input_data.copy()
        record["评估"] = result
        st.session_state.predictions.append(record)

        # 重置 AI 分析相关的会话状态
        st.session_state.ai_analysis_result = None
        st.session_state.messages = []
        st.session_state.show_ai_analysis = True
        # 不再要求用户输入API密钥
        st.session_state.api_key_entered = False
        if 'API_KEY' in st.session_state:
            del st.session_state.API_KEY
        if 'client' in st.session_state:
            del st.session_state.client  # 删除旧的 Ark 客户端


# 显示 AI 分析按钮
if st.button("开始 AI 分析"):
    # 显示 AI 分析部分
    st.subheader("AI 分析")
    st.info("生成潜在人因危害分析及改善建议：")
    # 直接使用预设的API密钥
    API_KEY = "5a5bd8a8-2257-4990-bac2-12b55ce17d4f"  # 直接设置 API_KEY
    if API_KEY:
        st.session_state.API_KEY = API_KEY
        st.session_state.api_key_entered = True
        # 初始化 Ark 客户端并存储在会话状态中
        try:
            st.session_state.client = Ark(api_key=API_KEY)  # 请确保 Ark 客户端正确初始化
        except Exception as e:
            st.error(f"初始化 Ark 客户端时出错：{e}")

     # AI 分析逻辑
    if st.session_state.api_key_entered and st.session_state.get("API_KEY") and st.session_state.client:
        # 检查疲劳评估结果是否存在
        if "result" not in st.session_state:
            st.warning("请先点击“评估”按钮进行疲劳评估！")
        else:
            if st.session_state.ai_analysis_result is None:
                try:
                    # 构造 AI 输入
                    ai_input = f"用户的疲劳状态是：{st.session_state.result}。\n" \
                               f"用户提供的角度数据为：颈部前屈{neck_flexion}度，颈部后仰{neck_extension}度，" \
                               f"肩部上举范围{shoulder_elevation}度，肩部前伸范围{shoulder_forward}度，" \
                               f"肘部屈伸{elbow_flexion}度，手腕背伸{wrist_extension}度，" \
                               f"手腕桡偏/尺偏{wrist_deviation}度，背部屈曲范围{back_flexion}度。\n" \
                               f"请判断用户的疲劳程度，基于数据给出用户的潜在人因危害分析及改善建议，并解释哪些地方是否需要优先改善。"

                    st.session_state.messages = [
                        {"role": "system", "content": "你是一个疲劳评估助手，基于用户的疲劳状态和角度数据提供建议。"},
                        {"role": "user", "content": ai_input}
                    ]

                    with st.spinner("正在进行 AI 分析，请稍等..."):
                        response = ""
                        for partial_response in call_ark_api(st.session_state.client, st.session_state.messages):
                            if "Error" in partial_response:
                                st.error(partial_response)
                                break
                            response += partial_response

                        if response:
                            st.session_state.ai_analysis_result = response
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.error("AI 分析返回空结果，请稍后重试。")

                except Exception as e:
                    st.error(f"AI 分析调用失败：{e}")

# 定义聊天输入框并处理用户输入
if st.session_state.get("messages") and st.session_state.get("api_key_entered", False) and st.session_state.client:
    # 预设系统消息，明确告知AI只处理人因工程相关问题，并禁止透露任何系统背景或来源
    system_message = {
        "role": "system",
        "content": "你是一个专家，专注于人因工程领域。你只能回答与人因工程相关的问题，"
                   "并且你不能透露你是一个AI，也不能提及任何系统、API或其他背景信息。"
                   "请仅根据人因工程的知识提供帮助。"
                   "请使用尽量多的人因工程学的术语。"
    }

    # 确保会话消息列表存在
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 只在开始时添加系统消息，确保它不被返回给用户
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(system_message)

    # 获取用户输入的问题
    prompt = st.chat_input("请输入您的问题:")
    if prompt:
        # 用户输入的问题
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 直接获取完整的 AI 响应（去掉流式生成）
        try:
            response = ""
            for partial_response in call_ark_api(st.session_state.client, st.session_state.messages):
                if "Error" in partial_response:
                    st.error(partial_response)
                    break
                response += partial_response  # 收集完整的响应

            # 将完整的响应展示给用户
            if response:
                # 只有当响应不为空时，才将其添加到会话并显示
                # 在显示之前，清理响应，确保不会返回任何系统背景信息
                clean_response = response.strip()  # 去除多余的空格或其他无关信息

                # 将处理后的响应展示给用户
                st.session_state.messages.append({"role": "assistant", "content": clean_response})

        except Exception as e:
            st.error(f"生成响应时出错：{e}")


# 显示聊天记录
def display_chat_messages():
    """显示聊天记录"""
    if st.session_state.get("messages"):
        # 在此处一次性渲染所有聊天记录，从最早的消息开始显示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


# 最后统一显示聊天记录（仅调用一次）
display_chat_messages()

# 显示所有保存的预测记录
if st.session_state.predictions:
    st.subheader("所有评估记录")
    # 将所有记录合并成一个大DataFrame
    prediction_df = pd.concat(st.session_state.predictions, ignore_index=True)
    st.write(prediction_df)
