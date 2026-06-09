import plotly.graph_objects as go
import plotly.express as px


def create_candlestick_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC"
    ))

    fig.update_layout(
        title="Candlestick Chart Bitcoin",
        template="plotly_dark",
        height=520,
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        xaxis_rangeslider_visible=False
    )

    return fig


def create_price_ma_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MA5"],
        mode="lines",
        name="MA5"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MA10"],
        mode="lines",
        name="MA10"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MA20"],
        mode="lines",
        name="MA20"
    ))

    fig.update_layout(
        title="Bitcoin Close Price dengan Moving Average",
        template="plotly_dark",
        height=520,
        xaxis_title="Tanggal",
        yaxis_title="Harga Bitcoin",
        legend_title="Indikator"
    )

    return fig


def create_rsi_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["RSI"],
        mode="lines",
        name="RSI"
    ))

    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold")

    fig.update_layout(
        title="Relative Strength Index / RSI",
        template="plotly_dark",
        height=420,
        xaxis_title="Tanggal",
        yaxis_title="RSI"
    )

    return fig


def create_macd_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD"],
        mode="lines",
        name="MACD"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD_Signal"],
        mode="lines",
        name="Signal Line"
    ))

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["MACD_Histogram"],
        name="MACD Histogram"
    ))

    fig.update_layout(
        title="Moving Average Convergence Divergence / MACD",
        template="plotly_dark",
        height=420,
        xaxis_title="Tanggal",
        yaxis_title="MACD"
    )

    return fig


def create_model_evaluation_chart(evaluation_df):
    chart_df = evaluation_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        chart_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        template="plotly_dark",
        title="Perbandingan Performa Model Machine Learning"
    )

    fig.update_layout(
        height=500,
        yaxis=dict(range=[0, 1])
    )

    return fig