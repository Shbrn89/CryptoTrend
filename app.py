import streamlit as st
import pandas as pd

from dataset_handler import load_and_clean_dataset, get_ohlcv_columns
from feature_engineering import (
    add_technical_indicators,
    get_feature_columns,
    get_market_condition
)
from ml_model import (
    create_classification_target,
    prepare_model_data,
    train_and_evaluate_models,
    predict_manual_ohlcv_input
)
from visualization import (
    create_candlestick_chart,
    create_price_ma_chart,
    create_rsi_chart,
    create_macd_chart,
    create_model_evaluation_chart
)


st.set_page_config(
    page_title="CryptoTrend",
    page_icon="₿",
    layout="wide"
)


def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


def show_metric_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction_box(label, confidence, explanation):
    if label == "Bullish":
        box_class = "bullish-box"
        color = "#22c55e"
    else:
        box_class = "bearish-box"
        color = "#ef4444"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="prediction-title" style="color:{color};">{label}</div>
            <div class="prediction-confidence">{confidence:.2f}% Probability</div>
            <div class="prediction-explain">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


load_css("style.css")


st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-logo">₿</div>
        <div>
            <div class="sidebar-title">CryptoTrend</div>
            <div class="sidebar-subtitle">Bitcoin ML Dashboard</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset Historis Bitcoin",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="sidebar-model-card">
        <div class="sidebar-model-label">Final Prediction Model</div>
        <div class="sidebar-model-value">Random Forest</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-info-card">
        <div class="sidebar-info-title">Workflow</div>
        <div class="sidebar-info-item">
            <span class="sidebar-dot"></span>
            <span>Dataset historis digunakan untuk training model.</span>
        </div>
        <div class="sidebar-info-item">
            <span class="sidebar-dot"></span>
            <span>Input OHLCV digunakan untuk prediksi Bullish/Bearish.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Machine Learning • Technical Indicator • Bitcoin</div>
        <div class="hero-title">CryptoTrend</div>
        <div class="hero-subtitle">
            Sistem klasifikasi tren pergerakan harga Bitcoin berbasis machine learning.
            Dataset historis digunakan untuk training Logistic Regression, SVM, dan Random Forest.
            Random Forest digunakan sebagai model final untuk memprediksi input OHLCV user menjadi Bullish atau Bearish.
        </div>
        <div class="flow-grid">
            <div class="flow-card">Historical Dataset</div>
            <div class="flow-card">OHLCV Cleaning</div>
            <div class="flow-card">MA • RSI • MACD</div>
            <div class="flow-card">Train ML Models</div>
            <div class="flow-card">Predict Bullish/Bearish</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


if uploaded_file is None:
    try:
        raw_df = pd.read_csv("Bitcoin_history_data.csv")
        dataset_status = "Dataset default berhasil dimuat."
    except FileNotFoundError:
        st.warning("Upload dataset Bitcoin terlebih dahulu melalui sidebar.")
        st.stop()
else:
    raw_df = pd.read_csv(uploaded_file)
    dataset_status = "Dataset berhasil diupload."


try:
    clean_df = load_and_clean_dataset(raw_df)
except ValueError as error:
    st.error(error)
    st.stop()


feature_df = add_technical_indicators(clean_df)
feature_columns = get_feature_columns()
target_df = create_classification_target(feature_df)

model_df = prepare_model_data(
    df=target_df,
    feature_columns=feature_columns
)

if len(model_df) < 100:
    st.error(
        "Data terlalu sedikit setelah preprocessing dan feature engineering. "
        "Gunakan dataset historis yang lebih panjang."
    )
    st.stop()


try:
    (
        evaluation_df,
        trained_models,
        comparison_best_model_name,
        final_model_name,
        final_model
    ) = train_and_evaluate_models(
        model_df=model_df,
        feature_columns=feature_columns
    )
except ValueError as error:
    st.error(error)
    st.stop()


latest_historical_row = target_df.dropna(subset=feature_columns).iloc[-1]
rsi_condition, macd_condition, trend_condition = get_market_condition(latest_historical_row)


dash_col1, dash_col2, dash_col3, dash_col4, dash_col5 = st.columns(5)

with dash_col1:
    show_metric_card(
        "Close Terakhir",
        f"${latest_historical_row['Close']:,.2f}",
        "Data historis terbaru"
    )

with dash_col2:
    show_metric_card(
        "RSI Terbaru",
        f"{latest_historical_row['RSI']:.2f}",
        rsi_condition
    )

with dash_col3:
    show_metric_card(
        "MACD Signal",
        macd_condition,
        f"MACD value: {latest_historical_row['MACD']:,.2f}"
    )

with dash_col4:
    show_metric_card(
        "Best Evaluation",
        comparison_best_model_name,
        "F1-Score & Accuracy"
    )

with dash_col5:
    show_metric_card(
        "Final Model",
        final_model_name,
        "Untuk prediksi akhir"
    )


dashboard_tab, prediction_tab, data_tab, indicator_tab, model_tab, feedback_tab = st.tabs([
    "Dashboard",
    "Prediction",
    "Dataset",
    "Technical Indicator",
    "Model Evaluation",
    "Feedback"
])


with dashboard_tab:
    left_col, right_col = st.columns([1.25, 1])

    with left_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Market Overview</div>
                <div class="section-desc">
                    Grafik ini menampilkan harga penutupan Bitcoin beserta indikator Moving Average.
                    Dashboard ini dibuat agar user langsung melihat kondisi pasar tanpa harus membaca banyak section panjang.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            create_price_ma_chart(target_df),
            use_container_width=True,
            key="dashboard_price_ma_chart"
        )

    with right_col:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">System Pipeline</div>
                <div class="section-desc">
                    Dataset historis Bitcoin dipakai untuk training model. 
                    Setelah itu user memasukkan data OHLCV terbaru, lalu Random Forest memprediksi tren berikutnya.
                </div>
                <div class="info-box">
                    <b>Model final:</b> {final_model_name}<br>
                    <b>Model pembanding terbaik:</b> {comparison_best_model_name}<br>
                    <b>Jumlah data training siap pakai:</b> {len(model_df)} baris<br>
                    <b>Status dataset:</b> {dataset_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.plotly_chart(
            create_rsi_chart(target_df),
            use_container_width=True,
            key="dashboard_rsi_chart"
        )


with prediction_tab:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Input OHLCV untuk Prediksi</div>
            <div class="section-desc">
                Masukkan data pasar terbaru Bitcoin. Sistem akan menggabungkan input ini dengan data historis,
                menghitung indikator teknikal terbaru, lalu Random Forest memprediksi hasil Bullish atau Bearish.
                Persentase yang ditampilkan berasal dari probability Random Forest.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        user_open = st.number_input(
            "Open",
            min_value=0.0,
            value=float(latest_historical_row["Open"]),
            step=100.0
        )

        user_high = st.number_input(
            "High",
            min_value=0.0,
            value=float(latest_historical_row["High"]),
            step=100.0
        )

    with input_col2:
        user_low = st.number_input(
            "Low",
            min_value=0.0,
            value=float(latest_historical_row["Low"]),
            step=100.0
        )

        user_close = st.number_input(
            "Close",
            min_value=0.0,
            value=float(latest_historical_row["Close"]),
            step=100.0
        )

    with input_col3:
        user_volume = st.number_input(
            "Volume",
            min_value=0.0,
            value=float(latest_historical_row["Volume"]),
            step=1000.0
        )

        st.write("")
        st.write("")
        predict_button = st.button("Predict Trend")

    if user_high < max(user_open, user_close, user_low):
        st.warning("Nilai High sebaiknya lebih besar atau sama dengan Open, Low, dan Close.")

    if user_low > min(user_open, user_close, user_high):
        st.warning("Nilai Low sebaiknya lebih kecil atau sama dengan Open, High, dan Close.")

    if predict_button:
        (
            prediction_label,
            confidence,
            user_latest_row,
            bullish_probability,
            bearish_probability
        ) = predict_manual_ohlcv_input(
            historical_df=clean_df,
            user_open=user_open,
            user_high=user_high,
            user_low=user_low,
            user_close=user_close,
            user_volume=user_volume,
            model=final_model,
            feature_columns=feature_columns,
            add_technical_indicators=add_technical_indicators
        )

        user_rsi_condition, user_macd_condition, user_trend_condition = get_market_condition(user_latest_row)

        if prediction_label == "Bullish":
            explanation = (
                "Model Random Forest memprediksi Bullish berdasarkan pola OHLCV dan indikator teknikal "
                "yang dipelajari dari dataset historis. Persentase diambil langsung dari probability model Random Forest."
            )
        else:
            explanation = (
                "Model Random Forest memprediksi Bearish berdasarkan pola OHLCV dan indikator teknikal "
                "yang dipelajari dari dataset historis. Persentase diambil langsung dari probability model Random Forest."
            )

        show_prediction_box(
            label=prediction_label,
            confidence=confidence,
            explanation=explanation
        )

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:
            st.metric("Bullish Probability", f"{bullish_probability:.2f}%")

        with prob_col2:
            st.metric("Bearish Probability", f"{bearish_probability:.2f}%")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric("RSI Input", f"{user_latest_row['RSI']:.2f}", user_rsi_condition)

        with result_col2:
            st.metric("MACD Input", f"{user_latest_row['MACD']:.2f}", user_macd_condition)

        with result_col3:
            st.metric("MA Trend Input", user_trend_condition)

    else:
        st.info("Isi data OHLCV lalu klik tombol Predict Trend.")


with data_tab:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Dataset & Preprocessing</div>
            <div class="section-desc">
                Dataset historis Bitcoin digunakan sebagai data training model. Sistem membersihkan data kosong,
                mengubah format tanggal, menyiapkan OHLCV, dan membuat estimasi Market Cap jika kolom tersebut tidak tersedia.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    data_col1, data_col2, data_col3 = st.columns(3)

    with data_col1:
        st.metric("Data Awal", len(raw_df))

    with data_col2:
        st.metric("Setelah Cleaning", len(clean_df))

    with data_col3:
        st.metric("Data Training Siap", len(model_df))

    st.subheader("Preview Dataset Awal")
    st.dataframe(raw_df.head(15), use_container_width=True)

    st.subheader("Data OHLCV Setelah Cleaning")
    st.dataframe(clean_df[get_ohlcv_columns()].head(20), use_container_width=True)

    st.subheader("Target Classification Preview")
    target_preview = target_df[["Date", "Close", "Next_Close", "Target"]].tail(20).copy()
    target_preview["Label"] = target_preview["Target"].map({
        1: "Bullish / Up",
        0: "Bearish / Down"
    })
    st.dataframe(target_preview, use_container_width=True)


with indicator_tab:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Technical Indicator Analysis</div>
            <div class="section-desc">
                Feature engineering dilakukan menggunakan MA5, MA10, MA20, RSI, dan MACD.
                Indikator ini membantu model membaca tren, momentum, dan volatilitas harga Bitcoin.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "Candlestick",
        "Moving Average",
        "RSI",
        "MACD"
    ])

    with chart_tab1:
        st.plotly_chart(
            create_candlestick_chart(target_df),
            use_container_width=True,
            key="indicator_candlestick_chart"
        )

    with chart_tab2:
        st.plotly_chart(
            create_price_ma_chart(target_df),
            use_container_width=True,
            key="indicator_price_ma_chart"
        )

    with chart_tab3:
        st.plotly_chart(
            create_rsi_chart(target_df),
            use_container_width=True,
            key="indicator_rsi_chart"
        )

    with chart_tab4:
        st.plotly_chart(
            create_macd_chart(target_df),
            use_container_width=True,
            key="indicator_macd_chart"
        )

    st.subheader("Technical Indicator Table")
    indicator_preview_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA5",
        "MA10",
        "MA20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram"
    ]
    st.dataframe(target_df[indicator_preview_columns].tail(25), use_container_width=True)


with model_tab:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Model Evaluation</div>
            <div class="section-desc">
                Logistic Regression dan SVM digunakan sebagai model pembanding.
                Random Forest digunakan sebagai model final sesuai proposal karena kuat terhadap data pasar kripto yang noisy dan non-linear.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        show_metric_card("Logistic Regression", "Baseline", "Pembanding awal")

    with model_col2:
        show_metric_card("SVM RBF", "Comparison", "Model non-linear")

    with model_col3:
        show_metric_card("Random Forest", "Final Model", "Dipakai untuk prediksi")

    display_evaluation = evaluation_df.copy()

    for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_evaluation[column] = display_evaluation[column].map(lambda value: f"{value:.4f}")

    st.subheader("Evaluation Table")
    st.dataframe(display_evaluation, use_container_width=True)

    st.plotly_chart(
        create_model_evaluation_chart(evaluation_df),
        use_container_width=True,
        key="model_evaluation_bar_chart"
    )

    st.markdown(
        f"""
        <div class="info-box">
            Berdasarkan evaluasi, model dengan performa tertinggi adalah <b>{comparison_best_model_name}</b>.
            Namun final prediction tetap menggunakan <b>{final_model_name}</b> agar sesuai dengan rancangan proposal.
            Model lain tetap ditampilkan sebagai pembanding.
        </div>
        """,
        unsafe_allow_html=True
    )


with feedback_tab:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">User Testing</div>
            <div class="section-desc">
                Form ini digunakan untuk mengumpulkan feedback pengguna terhadap usability aplikasi CryptoTrend.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("feedback_form"):
        user_name = st.text_input("Nama Pengguna")

        ease_score = st.slider("Kemudahan penggunaan aplikasi", 1, 5, 4)
        prediction_clarity_score = st.slider("Kejelasan hasil prediksi", 1, 5, 4)
        indicator_clarity_score = st.slider("Kejelasan tampilan indikator teknikal", 1, 5, 4)
        usefulness_score = st.slider("Kegunaan sistem bagi pengguna", 1, 5, 4)
        satisfaction_score = st.slider("Kepuasan pengguna secara keseluruhan", 1, 5, 4)

        comment = st.text_area("Komentar atau saran")

        submit_feedback = st.form_submit_button("Kirim Feedback")

        if submit_feedback:
            feedback_df = pd.DataFrame([{
                "Nama": user_name,
                "Kemudahan Penggunaan": ease_score,
                "Kejelasan Prediksi": prediction_clarity_score,
                "Kejelasan Indikator": indicator_clarity_score,
                "Kegunaan Sistem": usefulness_score,
                "Kepuasan": satisfaction_score,
                "Komentar": comment
            }])

            try:
                old_feedback = pd.read_csv("feedback_cryptotrend.csv")
                feedback_df = pd.concat([old_feedback, feedback_df], ignore_index=True)
            except FileNotFoundError:
                pass

            feedback_df.to_csv("feedback_cryptotrend.csv", index=False)

            st.success("Feedback berhasil disimpan. Terima kasih.")

    st.markdown(
        """
        <div class="disclaimer">
            Prediksi ini dibuat untuk tujuan edukasi dan penelitian.
            Hasil prediksi tidak boleh dianggap sebagai saran finansial atau rekomendasi investasi.
        </div>
        """,
        unsafe_allow_html=True
    )
