import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import torch.nn as nn
import joblib
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(layout="wide")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['axes.titlesize'] = 12  # 图标题字号
plt.rcParams['axes.labelsize'] = 6  # x/y轴标签字号
plt.rcParams['xtick.labelsize'] = 6  # x轴刻度字号
plt.rcParams['ytick.labelsize'] = 6  # y轴刻度字号
plt.rcParams['legend.fontsize'] = 6  # 图例字号


def aggregate_features(window):
    try:
        df_window = pd.DataFrame(window)
        numeric_data = df_window.select_dtypes(include=[np.number]).fillna(0)
        return np.concatenate([
            numeric_data.max(axis=0),
            numeric_data.min(axis=0),
            numeric_data.mean(axis=0)
        ])
    except:
        return np.zeros(261)


class DataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.samples = None
        self.labels = None
        self.timestamps = None
        self.patient_ids = None
        self.orig_feature_cols = []

    def load_data(self):
        self.orig_feature_cols = [
                                     c for c in self.df.columns
                                     if c not in ['检查时间', '序号', 'label']
                                 ] + ['入院天数']
        print("\n原始数据标签分布:")
        print(self.df['label'].value_counts(dropna=False).to_frame("数量"))
        return self.df

    def create_time_windows(self, window_size=3, median_mods_day=None):
        samples, labels, timestamps, patient_ids = [], [], [], []
        for pid in self.df['序号'].unique():
            try:
                patient_data = self.df[self.df['序号'] == pid].sort_values('检查时间')
                if len(patient_data) < window_size:
                    continue
                patient_data['检查时间'] = pd.to_datetime(patient_data['检查时间'])
                time_diff = patient_data['检查时间'] - patient_data['检查时间'].iloc[0]
                patient_data['入院天数'] = time_diff.dt.days + 1
                for i in range(len(patient_data) - window_size + 1):
                    window_end = i + window_size
                    current_label = patient_data.iloc[window_end - 1]['label']
                    if (median_mods_day is not None) and (current_label == 0):
                        days = patient_data.iloc[window_end - 1]['入院天数']
                        if days > median_mods_day:
                            continue
                    features = patient_data.iloc[i:window_end].drop(
                        columns=['检查时间', '序号', 'label'], errors='ignore'
                    ).fillna(0)
                    features = features.values.astype(np.float32)
                    samples.append(features)
                    labels.append(current_label)
                    timestamps.append(patient_data.iloc[window_end - 1]['检查时间'])
                    patient_ids.append(pid)
            except Exception as e:
                print(f"处理患者 {pid} 时出错: {str(e)}")
                continue
        self.samples = np.array(samples)
        self.labels = np.array(labels)
        self.timestamps = np.array(timestamps)
        self.patient_ids = np.array(patient_ids)
        return self.samples, self.labels, self.timestamps, self.patient_ids

    def prepare_xgboost_data(self):
        return np.array([aggregate_features(w) for w in self.samples])


# ==================== 模型定义 ====================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def load_models(model_dir, input_dim):
    xgb_model = joblib.load(model_dir / "xgb_model.pth")

    lstm = LSTMClassifier(input_dim=input_dim)
    lstm.load_state_dict(joblib.load(model_dir / "lstm_model.pth"))
    lstm.eval()

    gru = GRUClassifier(input_dim=input_dim)
    gru.load_state_dict(joblib.load(model_dir / "gru_model.pth"))
    gru.eval()

    return xgb_model, lstm, gru

def set_custom_style():
    custom_css = """
    <style>
        h1 { font-size: 20px !important; }
        h2 { font-size: 18px !important; }
        h3 { font-size: 16px !important; }
        p, label, div.stMarkdown, div.stTextInput > label, div.stNumberInput > label {
            font-size: 14px !important;
        }
        button, input, textarea, select {
            font-size: 14px !important;
        }
        table.dataframe th, table.dataframe td {
            font-size: 12px !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if __name__ == '__main__':
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'samples' not in st.session_state:
        st.session_state.samples = None
    if 'labels' not in st.session_state:
        st.session_state.labels = None
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = None
    if 'patient_ids' not in st.session_state:
        st.session_state.patient_ids = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    
    set_custom_style()
    
    col1, col2, col3 = st.columns([2, 2, 2])

    # ==================== col1：上传与预处理 ====================
    with col1:
        st.title("Prediction system for the incidence of MODS in severely burned patients")
        st.header("Based on XGBoost, LSTM, GRU and ensemble models")

        uploaded_file = st.file_uploader("请选择一个CSV或Excel文件", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("不支持的文件类型！")
                    st.stop()

                st.success("文件读取成功！以下是前五行数据：")
                st.dataframe(df.head())

                try:
                    processor = DataProcessor(df)
                    samples, labels, timestamps, patient_ids = processor.create_time_windows(window_size=3)

                    if len(samples) > 0:
                        st.subheader("预处理后的两个时间窗口样例")
                        sample_indices = []
                        seen_patients = set()
                        for i, pid in enumerate(patient_ids):
                            if pid not in seen_patients:
                                sample_indices.append(i)
                                seen_patients.add(pid)
                            if len(sample_indices) == 2:
                                break
                        while len(sample_indices) < 2 and len(sample_indices) < len(samples):
                            sample_indices.append(len(sample_indices))
                        for i in sample_indices:
                            st.markdown(f"🧪 样例窗口 {i + 1}（患者 ID：{patient_ids[i]}）")
                            st.dataframe(pd.DataFrame(samples[i]), use_container_width=True)

                        st.session_state.samples = samples
                        st.session_state.labels = labels
                        st.session_state.timestamps = timestamps
                        st.session_state.patient_ids = patient_ids
                        st.session_state.processor = processor
                        st.session_state.processing_done = True

                    else:
                        st.warning("未提取到有效的时间窗口样本，请检查是否包含 ['检查时间', '序号', 'label']")

                except Exception as e:
                    st.error(f"时间窗口预处理失败: {e}")

            except Exception as e:
                st.error(f"读取文件时出错：{e}")

    # ==================== col2：可视化分析 ====================
    with col2:
        if st.session_state.processing_done:
            st.markdown("### 📈 单个患者特征趋势图")
            unique_ids = list(np.unique(st.session_state.patient_ids))
            selected_features = ["血红蛋白浓度", "钾", "血糖", "白蛋白", "磷酸肌酸激酶同工酶", "白细胞"]

            df_raw = st.session_state.processor.df.copy()
            df_raw["检查时间"] = pd.to_datetime(df_raw["检查时间"])
            df_raw["入院天数"] = df_raw.groupby("序号")["检查时间"].transform(lambda x: (x - x.min()).dt.days + 1)

            patient_select = st.selectbox("选择一个患者查看趋势图", unique_ids)
            df_patient = df_raw[df_raw["序号"] == patient_select].sort_values("入院天数")

            fig1, ax1 = plt.subplots(figsize=(3, 3))
            for feat in selected_features:
                if feat in df_patient.columns:
                    ax1.plot(df_patient["入院天数"], df_patient[feat], marker="o", label=feat)
            ax1.set_title(f"患者 {patient_select} 特征趋势图")
            ax1.set_xlabel("入院天数")
            ax1.set_ylabel("数值")
            ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=4, framealpha=0.5)
            ax1.grid(True)
            st.pyplot(fig1)

            st.markdown("---")
            st.markdown("### 📊 两位患者关键特征平均值对比")
            col_a, col_b = st.columns(2)
            with col_a:
                patient_a = st.selectbox("患者 A", unique_ids, key="patient_a")
            with col_b:
                patient_b = st.selectbox("患者 B", unique_ids, key="patient_b")

            if st.button("🔍 执行对比"):
                try:
                    df1 = df_raw[df_raw["序号"] == patient_a]
                    df2 = df_raw[df_raw["序号"] == patient_b]

                    df1_clean = df1[selected_features].apply(pd.to_numeric, errors='coerce')
                    df2_clean = df2[selected_features].apply(pd.to_numeric, errors='coerce')

                    mean1 = df1_clean.mean()
                    mean2 = df2_clean.mean()

                    valid_mask = (~mean1.isna()) & (~mean2.isna())
                    mean1 = mean1[valid_mask]
                    mean2 = mean2[valid_mask]
                    filtered_features = mean1.index.tolist()

                    if len(filtered_features) == 0:
                        st.warning("⚠️ 无法绘图：两个患者在这些特征上都缺失数据。")
                    else:
                        fig2, ax2 = plt.subplots(figsize=(2, 2))
                        x = np.arange(len(filtered_features))
                        width = 0.35
                        ax2.bar(x - width / 2, mean1.values, width, label=f"患者 {patient_a}")
                        ax2.bar(x + width / 2, mean2.values, width, label=f"患者 {patient_b}")
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(filtered_features, rotation=45, fontsize=4)  # ✅ 缩小横坐标字体
                        ax2.set_ylabel("平均值", fontsize=4)  # ✅ 缩小坐标轴标题
                        ax2.set_title("关键特征平均值对比", fontsize=4)
                        ax2.legend()
                        ax2.grid(True)
                        st.pyplot(fig2)
                except Exception as e:
                    st.error(f"❌ 对比过程中出错：{e}")

    # ==================== col3：模型预测 ====================
    with col3:
        if st.session_state.processing_done:
            if st.button("🚀 执行模型预测"):
                with st.spinner("正在加载模型并进行预测..."):
                    model_dir = Path("./saved_models")
                    input_dim = st.session_state.samples[0].shape[1]
                    xgb_model, lstm_model, gru_model = load_models(model_dir, input_dim)

                    X_xgb = st.session_state.processor.prepare_xgboost_data()
                    xgb_probs = xgb_model.predict_proba(X_xgb)[:, 1]

                    X_tensor = torch.from_numpy(np.array(st.session_state.samples)).float()
                    with torch.no_grad():
                        lstm_probs = torch.sigmoid(lstm_model(X_tensor)).squeeze().numpy()
                        gru_probs = torch.sigmoid(gru_model(X_tensor)).squeeze().numpy()

                    ensemble1 = 0.5 * lstm_probs + 0.5 * xgb_probs
                    ensemble2 = 0.5 * gru_probs + 0.5 * xgb_probs

                    df_pred = pd.DataFrame({
                        "患者ID": st.session_state.patient_ids,
                        "时间": st.session_state.timestamps,
                        "XGBoost": xgb_probs,
                        "LSTM": lstm_probs,
                        "GRU": gru_probs,
                        "Ensemble1 (LSTM+XGB)": ensemble1,
                        "Ensemble2 (GRU+XGB)": ensemble2,
                    })

                    df_pred_sorted = df_pred.sort_values(by=["患者ID", "时间"])
                    st.subheader("🔍 模型预测结果")
                    float_cols = df_pred_sorted.select_dtypes(include=["float", "float64"]).columns
                    st.dataframe(df_pred_sorted.style.format({col: "{:.3f}" for col in float_cols}),
                                 use_container_width=True)
                    st.session_state.prediction_done = True

                st.markdown("### 📊 XGBoost 模型特征重要性（基于 Gain）")
                try:
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
                    xgb.plot_importance(
                        xgb_model,
                        ax=ax_imp,
                        importance_type='gain',  # 可选 'weight', 'gain', 'cover'
                        max_num_features=15,
                        height=0.4,
                        show_values=False
                    )
                    ax_imp.set_title("XGBoost 特征重要性（按 Gain 排序）")
                    plt.tight_layout()
                    st.pyplot(fig_imp)
                except Exception as e:
                    st.error(f"绘制特征重要性失败：{e}")

