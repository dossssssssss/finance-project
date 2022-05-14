import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import datetime
import json

with st.echo(code_location='below'):
    @st.cache
    def get_insiders_trading_info(ticker):
        url_finviz = "https://finviz.com/quote.ashx?t=" + ticker
        ### FROM: (https://www.adamsmith.haus/python/answers/how-to-set-the-user-agent-using-requests-in-python)
        headers = {"User-Agent": "aUserAgent"}

        ### END FROM:
        @st.cache
        def get_url_finviz():
            return requests.get(url_finviz, headers=headers)

        r_finviz = get_url_finviz()
        soup_finviz = BeautifulSoup(r_finviz.text)
        df_insiders_trading = pd.read_html(str(soup_finviz), attrs={"class": "body-table"})[0]
        df_insiders_trading = df_insiders_trading.drop([7, 8], axis=1)
        df_insiders_trading = df_insiders_trading.drop(0, axis=0)
        df_insiders_trading.rename(
            columns={0: "Insider Trader", 1: "Relationship", 2: "Date", 3: "Direction", 4: "Price, $",
                     5: "Number of shares", 6: "Total Value, $"}, inplace=True)

        # df_insiders_trading.count()[0]
        for i in range(1, int(df_insiders_trading.count()[0]) + 1):
            df_insiders_trading["Total Value, $"][i] = format(int(df_insiders_trading["Total Value, $"][i]), ",")
            df_insiders_trading["Number of shares"][i] = format(int(df_insiders_trading["Number of shares"][i]), ",")
        df_insiders_trading['Date'] = df_insiders_trading['Date'] + ", 2022"
        df_insiders_trading['Date'] = pd.to_datetime(df_insiders_trading['Date'], format="%b %d, %Y")
        # # dirty hack to get rid of 01 NOV of prev year
        first_date = df_insiders_trading['Date'][1]
        df_insiders_trading = df_insiders_trading[df_insiders_trading['Date'] <= first_date]
        return df_insiders_trading


    @st.cache
    def get_url_yahoo(ticker):
        url_yahoo_finance = "https://finance.yahoo.com/quote/" + ticker + "/history?p=" + ticker
        return requests.get(url_yahoo_finance, headers={"User-Agent": "aUserAgent"})


    @st.cache
    def get_prices_info(ticker):
        r_yahoo_finance = get_url_yahoo(ticker)
        soup_yahoo_finance = BeautifulSoup(r_yahoo_finance.text)
        df_prices = pd.read_html(str(soup_yahoo_finance), attrs={"class": "W(100%) M(0)"})[0]
        print(df_prices)
        for i in range(0, int(df_prices.count()[0])):
            if list(str(df_prices["Close*"][i]))[-1] == "d" or list(str(df_prices["Close*"][i]))[-1] == "t":
                df_prices = df_prices.drop([i], axis=0)

        df_prices.drop(df_prices.tail(2).index, inplace=True)  ###
        df_prices['Date'] = pd.to_datetime(df_prices['Date'], format="%b %d, %Y")
        for col in ['Close*']:
            df_prices[col] = df_prices[col].apply(pd.to_numeric)
        return df_prices


    @st.cache
    def get_url_marketwatch():
        url_marketwatch = "https://www.marketwatch.com/tools/screener/short-interest"
        return requests.get(url_marketwatch, headers={"User-Agent": "aUserAgent"})


    @st.cache
    def get_data_marketwatch():
        marketwatch = get_url_marketwatch()
        soup_marketwatch = BeautifulSoup(marketwatch.text)
        df_stocks = pd.read_html(str(soup_marketwatch))[0]
        df_stocks.rename(columns={df_stocks.columns[0]: "Symbol"}, inplace=True)
        df_stocks['Symbol'] = df_stocks['Symbol'].apply(lambda x: x.split(' ')[0])
        df_stocks["Price"] = pd.to_numeric(df_stocks["Price"].apply(lambda x: x[1:]))
        df_stocks["Chg% (1D)"] = pd.to_numeric(df_stocks["Chg% (1D)"].apply(lambda x: x[:-1]))
        df_stocks["Chg% (YTD)"] = pd.to_numeric(df_stocks["Chg% (YTD)"].apply(lambda x: x[:-1]))
        df_stocks["Float Shorted (%)"] = pd.to_numeric(df_stocks["Float Shorted (%)"].apply(lambda x: x[:-1]))
        return df_stocks


    @st.cache
    def get_url_yahoo_2(ticker):
        url_yahoo_finance_2 = "https://finance.yahoo.com/quote/" + ticker + "/analysis?p=" + ticker
        return requests.get(url_yahoo_finance_2, headers={"User-Agent": "aUserAgent"})


    def get_earnings_info(ticker):
        r_yahoo_finance_2 = get_url_yahoo_2(ticker)
        soup_yahoo_finance_2 = BeautifulSoup(r_yahoo_finance_2.text)
        df_earnings = \
        pd.read_html(str(soup_yahoo_finance_2), attrs={"class": "W(100%) M(0) BdB Bdc($seperatorColor) Mb(25px)"})[0]
        return df_earnings


    @st.cache
    def get_data_from_stockbeep():
        url = 'https://stockbeep.com/table-data/unusual-volume-stocks?country=us&time-zone=-180&sort-column=sstime&sort-order=desc'
        data = requests.get(url).text
        # soup = BeautifulSoup(res,'lxml')
        # table = soup.find_all('table')
        # st.write(table)
        data = json.loads(data)['data']
        df = pd.DataFrame(data)
        columns = {
            'sstime': 'Time', 'sscode': 'Ticker', 'ssname': 'Name',
            "sslast": "Last",
            "sshigh": "High",
            "sschg": "Chg",
            "sschgp": "Chg%",
            "ssvol": "Vol",
            "ssrvol": "RVol",
            "ss5mvol": "5mV",
            "sscap": "Cap",
            "sscomment": "Comment",
        }
        df.rename(columns=columns, inplace=True)
        df = df[list(columns.values())]
        df['Ticker'] = df['Ticker'].apply(lambda x: x[x.index('>') + 1:x.index('</')])
        df['Last'] = pd.to_numeric(df['Last'])
        df['Vol'] = df['Vol'].apply(lambda x: float(x[:-1]))
        df['Cap'] = df['Cap'].apply(lambda x: float(x[:-1]))
        df["High"] = pd.to_numeric(df['High'])
        df["Chg"] = pd.to_numeric(df['Chg'])
        df["Chg%"] = pd.to_numeric(df['Chg%'])
        df["RVol"] = pd.to_numeric(df['RVol'])
        df["RVol"] = pd.to_numeric(df['RVol'])

        return df


    tabs = ['Просмотр акций', "Поиск новинок"]
    current_tab = st.sidebar.radio('Режим работы', tabs)



    if current_tab == tabs[0]:
        st.subheader('Данные об акции: инсайдерская торговля и прогнозы аналитиков.')
        ticker = st.text_input("Введите тикер акции (например AAPL для Apple Inc. или MSFT для Microsoft Corporation).", key="name", value="AAPL")
        tick_yf = yf.Ticker(ticker)
        col1, col2 = st.columns([7, 20])
        with col1:
            st.image(tick_yf.info['logo_url'])
        with col2:
            st.write(f"***{tick_yf.info['shortName']}***")
            st.write(f"Индустрия: {tick_yf.info['industry']}")
            st.write(f"Сектор: {tick_yf.info['sector']}")

        st.subheader('Данные по инсайдерской торговле.')

        """
            На картинке ниже вы можете увидеть линейчатый график цены выбранной вами акции за последние 100 дней. 
            Кружки на графике соответствуют инсайдерским покупкам и продажам. Если навестись, то можно увидеть кто и именно совершил сделку и каков был объём.    
            
            Если инсайдеры покупают акцию, то может ли это означать что нечто хорошее ждёт компанию в ближайшем будущем?
        """

        prices_info = get_prices_info(ticker)[::-1]
        prices_info.sort_values(by='Date')
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=prices_info['Date'], y=prices_info['Close*'], name=ticker,
                       line=dict(color='royalblue', width=2)))

        insider_info = get_insiders_trading_info(ticker)
        insider_info = insider_info[insider_info['Direction'].isin(["Sale", 'Buy'])]
        insider_info.sort_values(by='Date')
        A = 5
        B = 35
        number_of_shares = [int(x.replace(',', '')) for x in insider_info['Number of shares']]
        a = min(number_of_shares)
        b = max(number_of_shares)
        K = (B - A) / (b - a)
        sizes = []
        texts = []
        colors = []
        for index, row in insider_info.iterrows():
            shares = int(row['Number of shares'].replace(',', ''))
            new_size = K * (shares - a) + A
            color = 'orangered' if row['Direction'] == 'Sale' else 'green'
            text = f"Инсайдер: {row['Insider Trader']}<br>Роль: {row['Relationship']}<br>Количество акций: {row['Number of shares']}<br>Стоимость акции: {row['Price, $']}"
            sizes.append(new_size)
            texts.append(text)
            colors.append(color)

        fig.add_trace(
            go.Scatter(x=insider_info['Date'], y=insider_info['Price, $'],
                       name="Insider",
                       # size="pop",
                       mode='markers',
                       text=texts,
                       marker=dict(
                           color=colors,
                           opacity=0.7,
                           size=sizes,
                       )
                       )
        )

        st.plotly_chart(fig)

        for key, value in tick_yf.info.items():
            if (key == "targetLowPrice"):
                min_prediction = value
            elif (key == "targetHighPrice"):
                max_prediction = value
            elif (key == "targetMeanPrice"):
                avg_prediction = value
            elif (key == "currentPrice"):
                current_price = value
            elif (key == "numberOfAnalystOpinions"):
                number_of_analytics = value

        st.subheader("Прогнозы аналитиков. ")

        """
            Ниже представлены текущая цена а также прогнозируемые цены. Точный список решил не приводить, но учтены мнения всех перечисленных на finance.yahoo.com аналитиков!    
        """

        fig = go.Figure()

        fig.add_shape(type="line",
                      x0=min_prediction, y0=1, x1=max_prediction, y1=1,
                      line=dict(
                          color="LightSeaGreen",
                          width=5,
                          dash="dashdot",
                      )
                      )

        fig.add_trace(
            go.Scatter(x=[min_prediction, avg_prediction, max_prediction], y=[1, 1, 1],
                       mode='markers',
                       text=[f'Минимум: {min_prediction:.2f}', f'Среднее: {avg_prediction:.2f}',
                             f'Максимальное: {max_prediction:.2f}'],
                       marker=dict(
                           color=['DarkOrange', 'DarkOrange', 'DarkOrange'],
                           opacity=1,
                           size=10
                       ),
                       showlegend=False)
        )

        fig.add_trace(
            go.Scatter(x=[min_prediction, avg_prediction, max_prediction], y=[1.2, 1.2, 1.2],
                       mode='text',
                       text=[f'MIN: {min_prediction:.2f}', f'AVG: {avg_prediction:.2f}', f'MAX: {max_prediction:.2f}'],
                       showlegend=False)
        )

        fig.add_trace(
            go.Scatter(x=[current_price], y=[1],
                       mode='markers',
                       text=[f'Текущая цена: {current_price:.2f}'],
                       marker=dict(
                           color=['LightSkyBlue'],
                           opacity=1,
                           size=10
                       ),
                       showlegend=False)
        )
        fig.add_trace(
            go.Scatter(x=[current_price], y=[1.4, 1.4, 1.4],
                       mode='text',
                       text=[f'CURRENT: {current_price:.2f}'],
                       showlegend=False)
        )

        fig.update_xaxes(range=[min_prediction - 30, max_prediction + 30], showgrid=False)
        fig.update_yaxes(range=[0, 2])
        col1, col2 = st.columns([3, 1])
        col1.plotly_chart(fig, use_container_width=True)
        diff = abs(min_prediction - current_price)
        if min_prediction < current_price:
            col2.write(f"Если сбудется **минимальный** прогноз - цена упадет на {100 * diff / current_price:.2f}%")
        else:
            col2.write(f"Если сбудется **минимальный** прогноз - цена вырастет на {100 * diff / current_price:.2f}%")
        diff = abs(avg_prediction - current_price)
        if avg_prediction < current_price:
            col2.write(f"Если сбудется **средний** прогноз - цена упадет на {100 * diff / current_price:.2f}%")
        else:
            col2.write(f"Если сбудется **средний** прогноз - цена вырастет на {100 * diff / current_price:.2f}%")
        diff = abs(max_prediction - current_price)
        if max_prediction < current_price:
            col2.write(f"Если сбудется **максимальный** прогноз - цена упадет на {100 * diff / current_price:.2f}%")
        else:
            col2.write(f"Если сбудется **максимальный** прогноз - цена вырастет на {100 * diff / current_price:.2f}%")

    elif current_tab == tabs[1]:
        # fig.show()

        st.subheader('Скринер акций: необычные объёмы и шорт-сквизы.')

        """
            Зачастую повышенные объёмы сигнализируют о повышенной волатильности. Возможно у фирмы вышел хороший отчёт в результате чего цена выросла? Или может наоброт? 
            В любом случае повышенные объёмы зачастую позволяют найти интересные акции к которым стоит присмотреться. Быть может следующая Tesla среди них?
            
            Снизу выберите ограничения по цене, рыночной капитализации и объёму акции. Далее наш сринер подберет вам акции с повышенным объёмом - в колонке Rvol вы можете найти во сколько раз он выше обычного.
        """
        # fig = px.line(prices_info, x="Date", y="Close*")

        stockbeep = get_data_from_stockbeep()
        min_price = min(stockbeep['Last'])
        max_price = max(stockbeep['Last'])
        min_v = min(stockbeep['Vol'])
        max_v = max(stockbeep['Vol'])
        min_c = min(stockbeep['Cap'])
        max_c = max(stockbeep['Cap'])
        start, end = st.slider("Цена (Last)", min_price, max_price, (min_price, max_price))
        start_c, end_c = st.select_slider("Капитализация (Cap)", sorted(stockbeep['Cap'].unique()), (min_c, max_c))
        start_v, end_v = st.select_slider("Объем (Vol)", sorted(stockbeep['Vol'].unique()), (min_v, max_v))
        n_rows = st.number_input("Сколько акций показать?", value=len(stockbeep))

        st.subheader('Данные об акциях с finance.yahoo.com')
        st.write(stockbeep[
                     (stockbeep['Last'] >= start) & (stockbeep['Last'] <= end) &
                     (stockbeep['Cap'] >= start_c) & (stockbeep['Cap'] <= end_c) &
                     (stockbeep['Vol'] >= start_v) & (stockbeep['Vol'] <= end_v)
                     ][:n_rows])

        select_value = st.selectbox(
            'Выберите признак для построения графика. Например, Last для последней цены акции или Rvol чтобы увидеть лидеров по относительному объёму.',
            (stockbeep.select_dtypes(include=np.number).columns.tolist()))

        x_1 = stockbeep['Ticker'].head(10).reset_index(drop=True).to_numpy()
        y_1 = stockbeep[stockbeep['Ticker'].isin(x_1)][select_value]

        graph_data_1 = pd.DataFrame({
            'Ticker': x_1,
            'Целевой признак': y_1
        })

        chart_1 = alt.Chart(graph_data_1).mark_line(
            point=alt.OverlayMarkDef(color="green")
        ).encode(
            x='Ticker',
            y='Целевой признак'
        ).properties(
            title='Данные об акциях с stockbeep.com'
        )
        st.altair_chart(chart_1, use_container_width=True)

        st.subheader(' ')

        """
            Данные на картинке ниже объедены по признаку того как повела себя цена в результате необычного объёма. 
            Например, 10M high обозначает что цена достигла наибольшего за последние 10 месяцев значения. Rct 52WL обозначает Revent 52 wekks low - недавно была зарегистрирована наименьшая цена за последние 52 недели.
            И так далее, больше об этих комментариях вы можете почитать на сайте stockbeep.com.
        """


        fig = plt.figure(figsize=(10, 4))
        fig = go.Figure(
            px.scatter(stockbeep, x="Ticker", y="RVol",
                       size="Cap", color="Comment", title=" ")
        )
        st.plotly_chart(fig, use_container_width=True)

        """
        Помимо необычных объёмов трейдеры следят за показателем Short Interest. Возможно в феврале 2021 года вы слышали о том
        как компании с Reddit'a удалось вызвать фурор на финансовом рынке, попасть в телевизор, заработать (а некоторым - разориться).
        Удивительно, но акции которыми они торговали имели достаточно высокий процент Float Shorted - если этот показатель высокий, то рост цены может 
        спровоцировать людей закрывать свои шорт-позиции и толкать цены ещё выше! 
        """

        marketwatch = get_data_marketwatch()
        st.subheader(' ')
        st.write(marketwatch)


        select_value_2 = st.selectbox(
            'Выберите признак для построения графика',
            (marketwatch.select_dtypes(include=np.number).columns.tolist()))

        x_2 = marketwatch['Symbol'].head(10).reset_index(drop=True).to_numpy()
        y_2 = marketwatch[marketwatch['Symbol'].isin(x_2)][select_value_2]

        graph_data_2 = pd.DataFrame({
            'Symbol': x_2,
            'Целевой признак': y_2
        })

        chart_2 = alt.Chart(graph_data_2).mark_line(
            point=alt.OverlayMarkDef(color="red")
        ).encode(
            x='Symbol',
            y='Целевой признак'
        ).properties(
            title='Данные marketwatch.com'
        )
        st.altair_chart(chart_2, use_container_width=True)