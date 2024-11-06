#!/usr/bin/env python
# coding: utf-8

# # Import thư viện



import warnings
warnings.filterwarnings('ignore')

import backtrader as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import tensorflow as tf
import tensorflow.keras.backend as K
import streamlit as st

from tensorflow.keras.layers import LSTM, Flatten, Dense, Masking
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from vnstock import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt

# Tải giá đóng cửa và thực hiện chiến thuật Trading SMA

list=listing_companies()
list=list[(list['organTypeCode']=='DN')&(list['comGroupCode']=='HOSE')]
mcp=list.ticker.to_list()

st.header("Ứng dụng mô hình học sâu để phân bổ danh mục đầu tư dựa trên chỉ báo kĩ thuật SMA")
st.subheader("Dữ liệu được chọn từ ngày 01/01/2018 đến 31/12/2023")

prices = st.file_uploader("Chọn file CSV để tải lên", type="csv")

# Kiểm tra nếu file đã được tải lên
if prices is not None:
    # Đọc file CSV bằng pandas
    prices = pd.read_csv(prices)

    # Hiển thị dữ liệu trong Streamlit
    st.write("Dữ liệu đã tải lên:")

    x=st.button("Bắt đầu tính toán", type="primary")
    if x==True:
        st.write("Đang thực hiện tính toán...")
        class Basic_MACrossStrategy(bt.Strategy):
            params = dict(ma_short_period=20, ma_long_period=50)

            def __init__(self):
                # Define the short-term (20-period) moving average
                self.ma_short = bt.indicators.MovingAverageSimple(self.data.close, period=self.p.ma_short_period, 
                                                                plotname='MA 20')

                # Define the long-term (50-period) moving average
                self.ma_long = bt.indicators.MovingAverageSimple(self.data.close, period=self.p.ma_long_period, 
                                                                plotname='MA 50')

                # Define the crossover signal (1 for upward cross, -1 for downward cross)
                self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)

            def next(self):
                # Buy when the short MA crosses above the long MA
                if self.crossover > 0 and not self.position:
                    self.buy(size=None)
                    print(f'BUY CREATE, {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}')

                # Sell when the short MA crosses below the long MA
                elif self.crossover < 0 and self.position:
                    self.sell(size=None)
                    print(f'SELL CREATE, {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}')


        R_ma_check=[]
        ticker_ma_check=[]
        for i in mcp:
            try:
                DT=prices[prices['ticker']==i]
                DT['time'] = pd.to_datetime(DT['time'])
                DT = DT.set_index('time')
                #TẠO DỮ LIỆU
                if len(DT)<1499:
                    continue
                DT[(DT.index >= '2018-01-01') & (DT.index <= '2022-12-31')]
                if len(DT)<1250:
                    continue
                data=bt.feeds.PandasData(dataname=DT)#DT LÀ DỮ LIÊU CỔ PHIẾU ĐÃ ĐƯỢC LẤY Ở TRÊN

                #thực thi chiến thuật
                cerebro=bt.Cerebro() #tạo cerebro

                cerebro.addstrategy(Basic_MACrossStrategy) #truyền chiến thuật


                cerebro.adddata(data) #truyền dữ liệu


                cerebro.broker.setcash(1000000000) #số tiền đầu tư
                cerebro.broker.setcommission(commission=0.0015) #số tiền hoa hồng/giao dịch
                cerebro.addsizer(bt.sizers.AllInSizerInt,percents = 95)#số cổ phiếu mua mỗi giao dịch


                print(i)
                before=cerebro.broker.getvalue()
                print('Số tiền trước khi thực hiện chiến thuật: %.2f' % before)
                cerebro.run() #thực thi chiến thuật
                after=cerebro.broker.getvalue()
                print('Số tiền sau khi thực hiện chiến thuật: %.2f' % after)
                r=(after-before)/before
                ticker_ma_check.append(i)
                R_ma_check.append(r)
            except Exception:
                continue
        return_ma_check=pd.DataFrame({'Ticker':ticker_ma_check,'Return':R_ma_check})

        return_ma_check=return_ma_check.sort_values('Return',ascending=False).head(50)

        mcp=return_ma_check.Ticker.to_list()

        list_allo=pd.DataFrame({'Asset':mcp})

        st.title('Biểu đồ thể hiện tỷ suất lợi nhuận của các cổ phiếu theo chỉ báo SMA')
        plt.figure(figsize=(20, 10)) 
        plt.bar(return_ma_check['Ticker'], return_ma_check['Return'],color='green')
        plt.xlabel('Mã cổ phiếu')
        plt.ylabel('Tỷ suất lợi nhuận')
        plt.xticks(rotation=45)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(plt)

        mcp=list_allo.Asset.to_list()


        # # Khai báo chiến thuật SMA

        # ### Ý tưởng chính là chia dữ liệu theo từng quý và chỉ tính lợi nhuận trong những khoảng thời gian mà có nắm giữ cổ phiếu (tức là chỉ khi đã mua cổ phiếu và trước khi bán).



        class MACrossStrategy(bt.Strategy):
            params = dict(ma_short_period=20, ma_long_period=50)

            def __init__(self):
                self.ma_short = bt.indicators.MovingAverageSimple(self.data.close, period=self.p.ma_short_period,
                                                                plotname='MA 20')
                self.ma_long = bt.indicators.MovingAverageSimple(self.data.close, period=self.p.ma_long_period,
                                                                plotname='MA 50')
                self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)
                self.last_order = None
                self.buy_price = None
                self.holding = False  # Trạng thái có nắm giữ cổ phiếu không
                self.current_quarter = None
                self.quarterly_returns = {}  # Lưu return theo từng quý

            def next(self):
                current_month = self.data.datetime.date(0).month
                current_year = self.data.datetime.date(0).year
                current_quarter = (current_year, (current_month - 1) // 3 + 1)  # Chia tháng theo quý

                if current_quarter not in self.quarterly_returns:
                    self.quarterly_returns[current_quarter] = 0

                # Mua cổ phiếu khi có tín hiệu
                if self.crossover > 0 and not self.position:
                    self.buy_price = self.data.close[0]
                    self.buy(size=None)
                    self.holding = True
                    self.current_quarter = current_quarter
                    print(f'BUY CREATE: {self.data.datetime.date(0)} - Buy price: {self.data.close[0]:.2f}')

                # Bán cổ phiếu khi có tín hiệu
                elif self.crossover < 0 and self.position:
                    sell_price = self.data.close[0]
                    self.sell(size=None)
                    profit_pct = (sell_price - self.buy_price) / self.buy_price
                    self.holding = False
                    self.quarterly_returns[self.current_quarter] += profit_pct
                    print(f'SELL CREATE: {self.data.datetime.date(0)} - Sell price: {self.data.close[0]:.2f}, Profit: {profit_pct:.2%}')

            def stop(self):
                if self.holding:
                    sell_price = self.data.close[0]
                    profit_pct = (sell_price - self.buy_price) / self.buy_price
                    self.quarterly_returns[self.current_quarter] += profit_pct
                    print(f'SELL ALL at the end: {self.data.datetime.date(0)} - Sell price: {self.data.close[0]:.2f}, Profit: {profit_pct:.2%}')



        # Khai báo biến lưu kết quả
        quarterly_returns_MA = {}

        for i in mcp:
            try:
                print(f"\nĐang xử lý mã: {i}")

                DT = stock_historical_data(i, '2018-01-01', '2023-12-31')
                DT['time'] = pd.to_datetime(DT['time'])
                DT = DT.reset_index()
                DT = DT.drop_duplicates(subset=['ticker', 'time'])
                DT = DT.set_index('time')

                data = bt.feeds.PandasData(dataname=DT)

                cerebro = bt.Cerebro()
                cerebro.addstrategy(MACrossStrategy)
                cerebro.adddata(data, name=i)
                cerebro.broker.setcash(1000000000)
                cerebro.broker.setcommission(commission=0.0015)
                cerebro.addsizer(bt.sizers.AllInSizerInt, percents=95)

                before = cerebro.broker.getvalue()
                print(f'Số tiền ban đầu: {before:.2f}')

                # Chạy chiến lược
                strategy_instances = cerebro.run()

                after = cerebro.broker.getvalue()
                print(f'Số tiền sau khi thực hiện chiến lược: {after:.2f}')

                # Tính tỷ lệ lợi nhuận
                r = (after - before) / before
                print(f'Lợi nhuận từ mã {i}: {r:.2%}')

                # Lưu lợi nhuận theo quý cho mã này
                quarterly_returns_MA[i] = strategy_instances[0].quarterly_returns

            except Exception as e:
                print(f"Error processing {i}: {e}")
                continue
                
        # Chuyển kết quả quarterly_returns_MA thành DataFrame
        quarterly_returns_df = pd.DataFrame.from_dict(quarterly_returns_MA, orient='index').T


        # Tạo tệp train từ năm 2018 đến 2022
        train_data = quarterly_returns_df.loc[quarterly_returns_df.index.get_level_values(0) <= 2022]
        # Tạo tệp test cho năm 2023
        test_data = quarterly_returns_df.loc[quarterly_returns_df.index.get_level_values(0) == 2023]
        # Reset index để đưa 'year' và 'quarter' về thành cột
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()


        # Xóa cột 'year' và 'quarter' sau khi reset index
        train_data = train_data.drop(columns=['level_0','level_1'])
        test_data = test_data.drop(columns=['level_0','level_1'])


        # Lớp CustomModel với hàm sharpe_loss
        class CustomModel:
            def __init__(self, data):
                self.data = data

            def sharpe_loss(self, _, y_pred):
                # Chia giá trị từng cột cho giá trị đầu tiên của cột đó
                data_normalized = tf.divide(self.data, self.data[0] + K.epsilon())
                # Tính giá trị danh mục đầu tư (portfolio)
                portfolio_values = tf.reduce_sum(tf.multiply(data_normalized, y_pred), axis=1)
                # Tránh chia cho 0 hoặc các giá trị bất thường
                portfolio_values = tf.where(tf.equal(portfolio_values, 0), K.epsilon(), portfolio_values)
                # Tính toán lợi nhuận danh mục đầu tư
                portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / (portfolio_values[:-1] + K.epsilon())
                # Tính Sharpe ratio
                sharpe = K.mean(portfolio_returns) / (K.std(portfolio_returns) + K.epsilon())
                return -sharpe

        X_train = train_data.values[np.newaxis, :, :]
        X_test = test_data.values[np.newaxis, :, :]
        y_train = np.zeros((1, len(train_data.columns)))
        y_test = np.zeros((1, len(test_data.columns)))


        # Khởi tạo mô hình tùy chỉnh
        data_tensor = tf.cast(tf.constant(train_data), float)
        custom_model = CustomModel(data_tensor)


        # Tạo mô hình LSTM
        model = Sequential([
            LSTM(1024, input_shape=train_data.shape),
            Flatten(),
            Dense(train_data.shape[1], activation='softmax')
        ])

        # Biên dịch mô hình
        model.compile(
            optimizer= 'Adam',
            loss=custom_model.sharpe_loss
        )


        model_LSTM = model.fit(X_train, y_train, epochs=100, shuffle=False)


        optimal_weights = model.predict(X_train)
        coeff_1 = optimal_weights[0]


        results_LSTM = pd.DataFrame({'Asset':mcp,"Weight":coeff_1})


        st.title('Biểu đồ phân bổ tài sản của danh mục đầu tư')
        square_plot_test = pd.DataFrame({
            'Cổ phiếu': results_LSTM.sort_values('Weight',ascending=False).Asset,
            'Tỷ trọng': results_LSTM.sort_values('Weight',ascending=False).Weight
        })
        square_plot_test.sort_values('Tỷ trọng',ascending=True)
        # Tạo nhãn mới bao gồm cả tên cổ phiếu và khối lượng
        square_plot_test['Nhãn'] = square_plot_test['Cổ phiếu'] + '\n' + square_plot_test['Tỷ trọng'].apply(lambda x: f"{x*100:.2f}").astype(str)+'%'

        # Định nghĩa màu sắc cho các khối vuông
        colors = ['#91DCEA', '#64CDCC', '#5FBB68', '#F9D23C', '#F9A729', '#FD6F30']


        # Vẽ biểu đồ khối vuông
        plt.figure(figsize=(15, 15))
        squarify.plot(sizes=square_plot_test['Tỷ trọng'], label=square_plot_test['Nhãn'], color=colors, alpha=.8, edgecolor='black', linewidth=2, text_kwargs={'fontsize':10})
        plt.axis('off')
        st.pyplot(plt)


        # # Tính toán E(r), Stddev và Sharpe ratio của danh mục và so sánh với allo 1 và allo 2

        import numpy as np

        def port_char(weights_df, returns_df):
            Er=returns_df.mean().reset_index()
            Er.columns=['Asset','Er']
            weights_df=pd.merge(weights_df,Er,on='Asset',how='outer')
            # Tính E(r) của portfolio
            portfolio_er = np.dot(weights_df['Weight'], weights_df['Er'])

            # Tính ma trận hiệp phương sai
            cov_matrix = returns_df.cov()

            # Tính độ lệch chuẩn của portfolio
            portfolio_std_dev = np.sqrt(np.dot(weights_df['Weight'].T, np.dot(cov_matrix, weights_df['Weight'])))

            return portfolio_er, portfolio_std_dev


        # Hàm tính toán tỷ lệ Sharpe của danh mục
        def sharpe_port(weights_df, returns_df):
            # Tính E(r) và std_dev của portfolio
            portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df)

            # Tính Sharpe Ratio
            sharpe_ratio = (portfolio_er -(0.016/(52*5)))/ portfolio_std_dev

            return sharpe_ratio


        Er,std_dev=port_char(results_LSTM,test_data)



        sharpe_LSTM=sharpe_port(results_LSTM,test_data)


        Allo_1=results_LSTM[['Asset']]

        Allo_1['Weight']=1/50

        Allo_2=train_data.sum().sort_values(ascending=False).reset_index()
        Allo_2.columns=['Asset','Er']


        Allo_2['Weight']=[0.8/(0.2*len(mcp))] * int(0.2*len(mcp)) + [0.2/(0.8*len(mcp))] * int(0.8*len(mcp))


        Allo_2=Allo_2[['Asset','Weight']]

        sharpe_port(Allo_1,test_data)

        sharpe_port(Allo_2,test_data)

        Er,std_dev=port_char(results_LSTM,test_data)
        Er_1,std_dev_1=port_char(Allo_1,test_data)
        Er_2,std_dev_2=port_char(Allo_2,test_data)
        table=pd.DataFrame({'Er':[Er,Er_1,Er_2],'Std_dev':[std_dev,std_dev_1,std_dev_2],
                            'Sharpe':[sharpe_port(results_LSTM,test_data),sharpe_port(Allo_1,test_data),sharpe_port(Allo_2,test_data)]})
        table=table.T
        table=table.rename(columns={0:'LSTM',1:'Allo_1',2:'Allo_2'})
        
        st.title('Hiệu quả hoạt động của danh mục LSTM và các danh mục truyền thống trong năm 2023')
        st.write(table)

        st.title('Biểu đồ thể hiện lợi nhuận trung bình và độ lệch chuẩn của các danh mục')
        plt.figure(figsize=(10, 6))
        categories=table.columns.values
        er_values=table.loc['Er'].values
        std_dev_values=table.loc['Std_dev'].values

        width = 0.2
        x_corr = np.arange(len(categories))
        plt.bar(x_corr,er_values,width,label='Er')
        plt.bar((x_corr+width),std_dev_values,width,label='std dev')

        plt.ylabel('Giá trị')
        plt.xticks((x_corr+width/2),categories)
        plt.legend()
        st.pyplot(plt)

        st.title('Biểu đồ thể hiện giá trị Sharpe của các danh mục')
        
        plt.figure(figsize=(10, 6))

        sharpe_values=table.loc['Sharpe'].values

        x_corr = np.arange(len(categories))
        plt.bar(x_corr,sharpe_values,width,label='Sharpe',color='green')

        plt.ylabel('Giá trị')
        plt.xticks((x_corr),categories)
        plt.legend()
        st.pyplot(plt)




