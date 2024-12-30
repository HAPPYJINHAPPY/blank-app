import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from matplotlib import font_manager
import os
import io
from volcenginesdkarkruntime import Ark

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

# 使用 Markdown 居中标题
st.markdown("<h1 style='text-align: center;'>疲劳评估测试系统</h1>", unsafe_allow_html=True)

st.sidebar.title("导航")
page = st.sidebar.radio("选择页面", ["疲劳评估", "批量评估", "分析可视化", "疲劳评估2"])

# Single prediction page
if page == "疲劳评估":
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
        shoulder_forward = st.slider("肩部前伸范围", 0, 90, 30)

    with col2:
        elbow_flexion = st.slider("肘部屈伸", 0, 180, 90)
        wrist_extension = st.slider("手腕背伸", 0, 90, 15)
        wrist_deviation = st.slider("手腕桡偏/尺偏", 0, 45, 10)
        back_flexion = st.slider("背部屈曲范围", 0, 90, 20)

    # Task parameters
    st.subheader("时间参数")
    col3, col4 = st.columns(2)
    with col3:
        task_duration = st.number_input("持续时间（秒）", min_value=0, value=6)
    with col4:
        movement_frequency = st.number_input("重复频率（每分钟）", min_value=0, value=5)

    # Input data aggregation
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
# 创建 Ark 客户端
    API_KEY = st.text_input("请输入 OpenAI API 密钥", type="password")
    if not API_KEY:
        st.info("请输入 OpenAI API 密钥以继续。", icon="🗝️")
    else:
        client = Ark(api_key=API_KEY)
        # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "predictions" not in st.session_state:
        st.session_state.predictions = []

    def display_chat_messages():
        """显示聊天记录"""
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])


    # 评估按钮
    result = None
    if st.button("评估"):
        with st.spinner("正在评估，请稍等..."):
            # 模型预测
            prediction = model.predict(input_data)
            result = ["低疲劳状态", "中疲劳状态", "高疲劳状态"][prediction[0]]
            st.success(f"评估结果：{result}")

            # 保存评估记录
            record = input_data.copy()
            record["评估"] = result
            st.session_state.predictions.append(record)

            # 自动分析的 AI 输入构造
            ai_input = f"用户的疲劳状态是：{result}。\n" \
                       f"用户提供的角度数据为：颈部前屈{neck_flexion}度，颈部后仰{neck_extension}度，" \
                       f"肩部上举范围{shoulder_elevation}度，肩部前伸范围{shoulder_forward}度，" \
                       f"肘部屈伸{elbow_flexion}度，手腕背伸{wrist_extension}度，" \
                       f"手腕桡偏/尺偏{wrist_deviation}度，背部屈曲范围{back_flexion}度。\n" \
                       f"请基于这些数据给出用户的潜在人因危害分析及改善建议。"

            # 将分析消息添加到聊天记录中
            st.session_state.messages.append({"role": "user", "content": ai_input})


            # 调用 Ark API 进行自动分析
            def call_ark_api(messages):
                try:
                    ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
                    completion = client.chat.completions.create(
                        model="ep-20241226165134-6lpqj",  # 使用正确的 Ark 模型ID
                        messages=ark_messages,
                        stream=True  # 流式响应
                    )

                    response = ""
                    for chunk in completion:
                        delta_content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta,
                                                                                  "content") else ""
                        yield delta_content
                except Exception as e:
                    st.error(f"调用 Ark API 时出错：{e}")
                    yield f"Error: {e}"


            # 创建占位符显示助手的回答
            response_placeholder = st.empty()
            response = ""  # 初始化完整响应

            # 仅使用流式响应更新聊天记录
            for partial_response in call_ark_api(st.session_state.messages):
                response += partial_response
                response_placeholder.markdown(response)  # 更新占位符内容

            # 只将流式生成的完整响应添加到聊天记录
            st.session_state.messages.append({"role": "assistant", "content": response})

        display_chat_messages()

    # 用户输入问题并获取答案
    if prompt := st.chat_input("请输入您的问题:"):
        # 仅在用户输入新问题时，将新问题追加到现有聊天记录中，而不清空聊天记录
        st.session_state.messages.append({"role": "user", "content": prompt})


        # 调用 Ark API 获取回答
        def call_ark_api_for_question(messages):
            try:
                ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
                completion = client.chat.completions.create(
                    model="ep-20241226165134-6lpqj",  # 使用正确的 Ark 模型ID
                    messages=ark_messages,
                    stream=True  # 流式响应
                )

                response = ""
                for chunk in completion:
                    delta_content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, "content") else ""
                    yield delta_content
            except Exception as e:
                st.error(f"调用 Ark API 时出错：{e}")
                yield f"Error: {e}"


        # 创建占位符来显示助手的回答
        response_placeholder = st.empty()
        response = ""
        for partial_response in call_ark_api_for_question(st.session_state.messages):
            response += partial_response
            response_placeholder.markdown(response)

        # 将助手的完整回答保存到会话状态
        st.session_state.messages.append({"role": "assistant", "content": response})

        display_chat_messages()

        # 将当前记录（包括输入数据和预测结果）添加到 session_state 中
        record = input_data.copy()
        record["评估"] = result
        st.session_state.predictions.append(record)

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # 显示SHAP特征贡献分析
        st.subheader("特征贡献分析")

        if isinstance(shap_values, list) and len(shap_values) > 1:
            st.write("SHAP values for Class 1")

            # SHAP summary plot
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
            st.pyplot(fig)  # 显示图形

        else:
            st.write("没有足够的SHAP值数据可用。")
    # 显示所有保存的预测记录
    if st.session_state.predictions:
        st.subheader("所有评估记录")
        # 将所有记录合并成一个大DataFrame
        prediction_df = pd.concat(st.session_state.predictions, ignore_index=True)
        st.write(prediction_df)
    if st.button("生成雷达图"):
        def plot_radar_chart(features, values, title="Radar Chart"):
            num_vars = len(features)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            values += values[:1]
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='blue', alpha=0.25)
            ax.plot(angles, values, color='blue', linewidth=2)
            ax.set_yticks([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features, fontsize=10)
            ax.set_title(title, fontsize=14, pad=20)
            set_font_properties(ax, font_prop)
            return fig


        radar_fig = plot_radar_chart(input_data.columns.tolist(), input_data.iloc[0].values.tolist())
        st.pyplot(radar_fig)

# Batch prediction page
elif page == "批量评估":
    st.title("批量评估")
    uploaded_file = st.file_uploader("上传csv文件进行批量评估", type="csv")

    if uploaded_file is not None:
        st.success("上传文件成功")
        data = pd.read_csv(uploaded_file, encoding='gbk')
        st.dataframe(data.head())

        # 检查是否包含目标变量
        if '疲劳等级' in data.columns:
            st.warning(
                "The uploaded file includes the target variable 'Fatigue_Label'. It will be excluded from prediction.")
            X = data.drop(columns=['疲劳等级'])
        else:
            X = data  # 默认使用上传数据的所有列

        # 检查特征数量是否匹配
        expected_features = len(model.feature_importances_)
        if len(X.columns) == expected_features:
            predictions = model.predict(X)
            data['疲劳值'] = predictions
            st.subheader("评估结果")
            st.write(data)

            # 特征重要性
            feature_importances = pd.DataFrame({
                'Feature': X.columns[:expected_features],  # 动态匹配特征列
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            st.subheader("特征重要性")
            st.write(feature_importances)

            # 显示重要特征
            important_feature_1 = feature_importances.iloc[0]['Feature']
            important_feature_2 = feature_importances.iloc[1]['Feature']

            st.markdown(f"### Top Features: `{important_feature_1}` and `{important_feature_2}`")

            # 绘制散点图
            st.subheader("特征散点图")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=data,
                x=important_feature_1,
                y=important_feature_2,
                hue="疲劳值",
                palette="coolwarm",
                s=100
            )
            plt.title(f"散点图{important_feature_1} vs {important_feature_2}", fontproperties=font_prop)
            plt.xlabel(important_feature_1, fontproperties=font_prop)
            plt.ylabel(important_feature_2, fontproperties=font_prop)
            st.pyplot(plt)

            # 散点矩阵（Pairplot）显示主要特征间的关系
            st.subheader("疲劳等级间主要特征的散点矩阵")

            # 用户选择用于绘图的特征
            top_features = st.multiselect(
                "选择主要特征（最多选择5个）",
                options=X.columns.tolist(),
                default=X.columns[:3].tolist()  # 默认选择前三个特征
            )

            if top_features:
                # 限制选择特征的数量，避免过多特征导致图形过于复杂
                if len(top_features) > 5:
                    st.warning("为了更好的可视化效果，请选择不超过5个特征。")
                else:
                    top_features_with_label = top_features + ['疲劳值']

                    try:
                        # 绘制散点矩阵
                        fig = sns.pairplot(
                            data[top_features_with_label],
                            hue="疲劳值",
                            palette="husl",
                            diag_kind="kde"
                        )
                        fig.fig.suptitle("主要特征与疲劳预测结果的散点矩阵", y=1.02, fontproperties=font_prop)
                        for ax in fig.axes.flatten():
                            set_font_properties(ax, font_prop)

                        st.pyplot(fig)
                    except ValueError as e:
                        st.error(f"绘制失败，请检查所选特征和数据的一致性：{e}")
            # 绘制相关性热图
            st.subheader("相关性热图")
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            set_font_properties(ax, font_prop)
            st.pyplot(fig)

            # 提供下载按钮
            st.download_button(
                label="下载评估结果",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv',
            )
        else:
            st.error(
                f"Feature mismatch: Model expects {expected_features} features, but the uploaded file contains {len(X.columns)}.")
elif page == "分析可视化":

    # 页面标题
    st.title("自定义数据可视化仪表板")

    # 上传文件功能
    uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx"])

    # 读取上传的文件并处理
    if uploaded_file:
        try:
            if uploaded_file.name.endswith("csv"):
                # 尝试读取CSV文件，指定常见编码格式
                try:
                    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('ISO-8859-1')))
            else:
                # 读取Excel文件
                df = pd.read_excel(uploaded_file)

            # 显示数据表
            st.write("数据预览：")
            st.write(df.head())

        except Exception as e:
            # 捕获任何读取错误并提示用户
            st.error(f"文件读取失败: {e}")

        # 获取文件的列名
        columns = df.columns.tolist()
        st.write("数据列名：", columns)

        # 用户选择分析的列
        selected_columns = st.multiselect("请选择要分析的列", columns)

        if selected_columns:
            # 图表自定义选项
            st.sidebar.header("图表自定义")

            # 图表标题
            chart_title = st.sidebar.text_input("图表标题", "默认图表标题")

            # 图表颜色选择
            color = st.sidebar.color_picker("选择图表颜色", "#1f77b4")

            # 区分单变量和多变量图表
            is_single_variable = len(selected_columns) == 1
            is_multiple_variables = len(selected_columns) > 1

            # 单变量图表选择
            if is_single_variable:
                chart_types = st.sidebar.multiselect(
                    "选择图表类型 (单变量)", ["趋势图", "直方图", "箱线图"]
                )

                if "趋势图" in chart_types:
                    st.subheader(f"{chart_title} - 趋势图")
                    fig = go.Figure(
                        go.Scatter(x=df.index, y=df[selected_columns[0]], mode='lines+markers', line=dict(color=color)))
                    fig.update_layout(title=chart_title)
                    st.plotly_chart(fig)

                if "直方图" in chart_types:
                    st.subheader(f"{chart_title} - 直方图")
                    fig = px.histogram(df, x=selected_columns[0])
                    st.plotly_chart(fig)

                if "箱线图" in chart_types:
                    st.subheader(f"{chart_title} - 箱线图")
                    fig = px.box(df, y=selected_columns[0])
                    st.plotly_chart(fig)

            # 多变量图表选择
            elif is_multiple_variables:
                chart_types = st.sidebar.multiselect(
                    "选择图表类型 (多变量)", ["散点图", "热力图", "面包屑图"]
                )

                if "散点图" in chart_types:
                    st.subheader(f"{chart_title} - 散点图")
                    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1])
                    st.plotly_chart(fig)

                if "热力图" in chart_types:
                    st.subheader(f"{chart_title} - 热力图")
                    fig = go.Figure(data=go.Heatmap(
                        z=df[selected_columns].corr().values,  # 相关性矩阵
                        x=selected_columns,
                        y=selected_columns,
                        colorscale="Viridis"
                    ))
                    fig.update_layout(title="热力图")
                    st.plotly_chart(fig)

                if "面包屑图" in chart_types:
                    st.subheader(f"{chart_title} - 面包屑图")
                    if len(selected_columns) >= 3:
                        fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], size=selected_columns[2],
                                         hover_data=df.columns)
                        st.plotly_chart(fig)

        else:
            st.write("请选择要分析的列")

    else:
        st.write("请上传数据文件以开始分析。")
elif page == "疲劳评估2":

    # 定义疲劳状态判断规则
    def check_fatigue_status(score):
        if score < 20:
            return "正常状态"
        elif 20 <= score < 40:
            return "预警状态"
        else:
            return "高危状态"


    # Streamlit 应用布局
    st.title("疲劳分数监测与警报系统")

    # 上传文件功能
    uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx"])

    # 读取上传的文件并处理
    if uploaded_file:
        # 根据上传的文件类型读取数据
        if uploaded_file.name.endswith("csv"):
            daily_stats = pd.read_csv(uploaded_file)
        else:
            daily_stats = pd.read_excel(uploaded_file)

        # 确保数据包含必要的列
        required_columns = {'Day', 'Neck_Angle', 'Shoulder_Raise_Angle'}
        if not required_columns.issubset(daily_stats.columns):
            st.error(f"数据文件缺少必要的列：{', '.join(required_columns - set(daily_stats.columns))}")
        else:
            # 按天计算均值和标准差
            daily_stats_grouped = daily_stats.groupby('Day').agg(
                Neck_Angle_mean=('Neck_Angle', 'mean'),
                Shoulder_Raise_Angle_mean=('Shoulder_Raise_Angle', 'mean'),
                Neck_Angle_std=('Neck_Angle', 'std'),
                Shoulder_Raise_Angle_std=('Shoulder_Raise_Angle', 'std')
            ).reset_index()

            # 计算疲劳分数
            daily_stats_grouped['Fatigue_Score'] = (
                    daily_stats_grouped['Neck_Angle_mean'] * 0.4 +
                    daily_stats_grouped['Shoulder_Raise_Angle_mean'] * 0.4 +
                    daily_stats_grouped['Neck_Angle_std'] * 0.1 +
                    daily_stats_grouped['Shoulder_Raise_Angle_std'] * 0.1
            ).round(4)

            # 添加疲劳状态列
            daily_stats_grouped['Fatigue_Status'] = daily_stats_grouped['Fatigue_Score'].apply(check_fatigue_status)

            # 显示每日疲劳分数
            st.subheader("每日疲劳分数")
            st.dataframe(daily_stats_grouped)

            # 绘制疲劳分数趋势图
            st.subheader("疲劳分数趋势图")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(daily_stats_grouped['Day'], daily_stats_grouped['Fatigue_Score'], marker='o', label="疲劳分数",
                    color='b')
            ax.set_xticks(np.arange(1, len(daily_stats_grouped['Day']) + 1, 1))
            ax.set_xticklabels(np.arange(1, len(daily_stats_grouped['Day']) + 1, 1), fontsize=12)
            ax.set_title("每日疲劳分数趋势", fontsize=16)
            ax.set_xlabel("天数", fontsize=12)
            ax.set_ylabel("疲劳分数", fontsize=12)
            ax.legend()
            set_font_properties(ax, font_prop)
            st.pyplot(fig)

            # 触发警报
            st.subheader("疲劳状态警报")
            for _, row in daily_stats_grouped.iterrows():
                if row['Fatigue_Status'] == "预警状态":
                    st.warning(f"第 {row['Day']} 天：疲劳分数为 {row['Fatigue_Score']}，需要注意！")
                elif row['Fatigue_Status'] == "高危状态":
                    st.error(f"第 {row['Day']} 天：疲劳分数为 {row['Fatigue_Score']}，警报！立即调整任务或休息！")
                else:
                    st.success(f"第 {row['Day']} 天：疲劳分数为 {row['Fatigue_Score']}，状态正常。")
            # 增加天数间分布对比图
            st.subheader("所有天数之间的分布对比图")

            # 颈部角度分布对比图（所有天数）
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=daily_stats, x='Day', y='Neck_Angle', ax=ax)
            ax.set_title("各天颈部角度分布对比", fontsize=16)
            ax.set_xlabel("天数", fontsize=12)
            ax.set_ylabel("颈部角度（度）", fontsize=12)
            set_font_properties(ax, font_prop)
            st.pyplot(fig)

            # 肩部抬高角度分布对比图（所有天数）
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=daily_stats, x='Day', y='Shoulder_Raise_Angle', ax=ax)
            ax.set_title("各天肩部抬高角度分布对比", fontsize=16)
            ax.set_xlabel("天数", fontsize=12)
            ax.set_ylabel("肩部抬高角度（度）", fontsize=12)
            set_font_properties(ax, font_prop)
            st.pyplot(fig)

            # 绘制每一天的角度分布（直方图）
            st.subheader("每一天角度的分布图（直方图）")
            # 颈部角度每一天的分布
            fig, ax = plt.subplots(figsize=(10, 6))
            for day in daily_stats['Day'].unique():
                sns.kdeplot(daily_stats[daily_stats['Day'] == day]['Neck_Angle'], label=f"第 {day} 天", ax=ax)
            ax.set_title("每日颈部角度分布对比", fontsize=16)
            ax.set_xlabel("角度（度）", fontsize=12)
            ax.set_ylabel("密度", fontsize=12)
            ax.legend()
            # 统一设置字体
            set_font_properties(ax, font_prop)
            st.pyplot(fig)

            # 肩部抬高角度每一天的分布
            fig, ax = plt.subplots(figsize=(10, 6))
            for day in daily_stats['Day'].unique():
                sns.histplot(daily_stats[daily_stats['Day'] == day]['Shoulder_Raise_Angle'], kde=True,
                             label=f"第 {day} 天", ax=ax)
            ax.set_title("每日肩部抬高角度分布对比", fontsize=16)
            ax.set_xlabel("角度（度）", fontsize=12)
            ax.set_ylabel("频次", fontsize=12)
            ax.legend()
            set_font_properties(ax, font_prop)
            st.pyplot(fig)
            # 创建单天疲劳分析区域
            st.subheader("单天疲劳分析")
            selected_day = st.selectbox("选择分析的天数", daily_stats_grouped['Day'].tolist())
            day_data = daily_stats[daily_stats['Day'] == selected_day]
            day_stats = daily_stats_grouped[daily_stats_grouped['Day'] == selected_day].iloc[0]

            # 计算该天的疲劳分数
            day_fatigue_score = (
                    day_stats['Neck_Angle_mean'] * 0.4 +
                    day_stats['Shoulder_Raise_Angle_mean'] * 0.4 +
                    day_stats['Neck_Angle_std'] * 0.1 +
                    day_stats['Shoulder_Raise_Angle_std'] * 0.1
            )

            st.write(f"选择的第 {selected_day} 天的疲劳分数为: {day_fatigue_score:.2f}")
            day_fatigue_status = check_fatigue_status(day_fatigue_score)
            st.write(f"疲劳状态: {day_fatigue_status}")

            # 绘制该天的数据分布图
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(day_data['Neck_Angle'], kde=True, label="颈部角度分布", color='blue', ax=ax)
            sns.histplot(day_data['Shoulder_Raise_Angle'], kde=True, label="肩部抬高角度分布", color='orange', ax=ax)
            ax.set_title(f"第 {selected_day} 天数据分布", fontsize=16)
            ax.set_xlabel("角度（度）", fontsize=12)
            ax.set_ylabel("频次", fontsize=12)
            ax.legend()
            set_font_properties(ax, font_prop)
            st.pyplot(fig)

            # 画出该天的疲劳评分趋势（若数据为时间序列数据）
            if 'Time' in day_data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(day_data['Time'], day_data['Neck_Angle'], label="颈部角度")
                ax.plot(day_data['Time'], day_data['Shoulder_Raise_Angle'], label="肩部抬高角度")
                ax.set_title(f"第 {selected_day} 天的疲劳评分趋势", fontsize=16)
                ax.set_xlabel("时间", fontsize=12)
                ax.set_ylabel("角度（度）", fontsize=12)
                ax.legend()
                set_font_properties(ax, font_prop)
                st.pyplot(fig)
    else:
        st.info("请上传一个包含以下列的 CSV 文件：Day, Neck_Angle, Shoulder_Raise_Angle")
