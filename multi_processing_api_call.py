import multiprocessing
import requests
import time
import sqlite3
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime


def perform_calc(df, symbol, connection):
    #gett all the past data. 
    # Use a SQL query to select all rows from the table
    query = f"SELECT * FROM {symbol};"
    
    # Use pandas read_sql_query to load the table into a DataFrame
    past_all_df = pd.read_sql_query(query, connection)

    past_all_df['Expiry_Date'] = pd.to_datetime(past_all_df['Expiry_Date'], format='%Y-%m-%d')

    # prev_time = past_all_df['Time'].min()
    # prev_df = past_all_df[past_all_df['Time']==prev_time]

    #seprating only the data of last cycle as calc depend on last cycle
    prev_df=past_all_df.tail(len(df))

    #sorting the values as the the difrence of corresponding columns is to be calculated
    prev_df = prev_df.sort_values(by=['Expiry_Date', 'Strike_Price'])
    df = df.sort_values(by=['Expiry_Date', 'Strike_Price'])

    #resetting indexes from 0 as the difrence happens wrt to index
    prev_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    #performing calc
    df['C_Calls'] = df['COI_Calls'] - prev_df['COI_Calls']
    df['C_Puts'] = df['COI_Puts'] - prev_df['COI_Puts']
    df['C_Amt_Calls_Cr'] = (df['C_Calls']*132000)/10000000
    df['C_Amt_Puts_Cr'] = (df['C_Puts']*132000)/10000000
    df['S_C_Calls'] = df.groupby('Expiry_Date')['C_Calls'].transform('sum')
    df['S_C_Puts'] = df.groupby('Expiry_Date')['C_Puts'].transform('sum')
    df['S_COI_Calls'] = df.groupby('Expiry_Date')['COI_Calls'].transform('sum')
    df['S_COI_Puts'] = df.groupby('Expiry_Date')['COI_Puts'].transform('sum')
    df['R_S_COI'] = np.where(df["S_COI_Calls"] != 0, df['S_COI_Puts'] / df["S_COI_Calls"], np.nan)

    #returning the current cycle data with columns of calc 
    return df




def call_api(symbol, headers, cookies):
    connection = sqlite3.connect("stockOptionChainData.db")
    cursor = connection.cursor()

    if symbol in ["NIFTY", "FINNIFTY", "BANKNIFTY", "MIDCPNIFTY"]:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        symbol = re.sub(r'[^a-zA-Z0-9]', '_', symbol) #to avoid special characters in table names
    response = requests.get(url, headers = headers, cookies=cookies)


    if response.status_code == 200:
        data = response.text
        data = json.loads(data)
        if data == {}:
            return False
        data = data['records']
        # strikePrices = data['strikePrices']
        # expiryDates = data['expiryDates']
        underlyingValue = data['underlyingValue']
        symbol_underlyingValues = [value[0] for value in cursor.execute("SELECT symbol FROM initialUnderlyingValues;").fetchall()]

        if symbol not in symbol_underlyingValues:
            # Insert data into the table
            cursor.execute('''
                INSERT INTO initialUnderlyingValues (symbol, underlyingValue)
                VALUES (?, ?)
            ''', [symbol, underlyingValue])

            # Commit the changes and close the connection
            connection.commit()
            initialUnderlyingValue = underlyingValue
        else:
                # Execute a SELECT query to retrieve the underlyingValue for the given symbol
                cursor.execute("SELECT underlyingValue FROM initialUnderlyingValues WHERE symbol = ?;", (symbol,))

                # Check if the result is not None and return the underlyingValue
                initialUnderlyingValue = cursor.fetchone()[0]


            
        timestamp = data['timestamp'][-8:]

        relevant_data = data['data']
        extracted_data = []

        for i in relevant_data:
            record = [timestamp, underlyingValue, i['strikePrice']]
            try:
                record.append(i['CE']['changeinOpenInterest'])
            except:
                record.append(0)
            try:
                record.append(i['CE']['totalBuyQuantity'])
            except:
                record.append(0)
            try:
                record.append(i['CE']['totalSellQuantity'])
            except:
                record.append(0)
            try:
                record.append(i['PE']['changeinOpenInterest'])
            except:
                record.append(0)
            try:
                record.append(i['PE']['totalBuyQuantity'])
            except:
                record.append(0)
            try:
                record.append(i['PE']['totalSellQuantity'])
            except:
                record.append(0)
            record.append(i['expiryDate'])
            extracted_data.append(record)

        #create df from extracted records with the desired field names
        df = pd.DataFrame(extracted_data, columns = ['Time', 'underlyingValue', 'Strike_Price', 'COI_Calls', 'Total_Buy_Calls', 'Total_Sell_Calls', 'COI_Puts', 'Total_Buy_Puts', 'Total_Sell_Puts', 'Expiry_Date'])

        #convert the expiry date fro string to datetime format so as to sort according to the dates. 
        df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], format='%d-%b-%Y')

        #sort the data frame first by expiry date and then by strike price
        df = df.sort_values(by=['Expiry_Date', 'Strike_Price'])


        #setting 12 above and below each strike prices of symbol if not set

        # Filter data based on underlyingValue (max 12 above and belo each)
        above_df = df[df['Strike_Price'] > initialUnderlyingValue].groupby('Expiry_Date').head(20)
        below_df = df[df['Strike_Price'] <= initialUnderlyingValue].groupby('Expiry_Date').tail(20)

        # Concatenate the results
        result_df = pd.concat([above_df, below_df], ignore_index=True)



        #sort the filter data 
        result_df = result_df.sort_values(by=['Expiry_Date', 'Strike_Price'])

        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}';")
        
        # Check if the query returned any rows
        result = cursor.fetchone()

        if result is None:
                #so as to add all the fields
                result_df['C_Calls'] = 0
                result_df['C_Puts'] = 0
                result_df['C_Amt_Calls_Cr'] = 0.0
                result_df['C_Amt_Puts_Cr'] = 0.0
                result_df['S_C_Calls'] = 0
                result_df['S_C_Puts'] = 0
                result_df['S_COI_Calls'] = 0
                result_df['S_COI_Puts'] = 0
                result_df['R_S_COI'] = 0.0
        else:
            result_df = perform_calc(result_df, symbol, connection)

        #convert expiry date from datetime back to sting date to as to maintain a uniformity while reading and combining data from csv while later on.
        result_df['Expiry_Date'] = result_df['Expiry_Date'].dt.strftime('%Y-%m-%d')

        result_df.to_sql(name=symbol, con=connection, index=False, if_exists='append')

        cursor.close()
        connection.close()

        return True
    
    return False
        

        
def call_batch(symbols):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36','Accept-Encoding': 'gzip, deflate, br','Accept-Language': 'en-US,en;q=0.9,hi;q=0.8'}
        response = requests.get("https://nseindia.com/option-chain", headers=headers)
        if response.status_code == 200:
            cookies = response.cookies
            results = []
            for symbol in symbols:
                results.append(call_api(symbol, headers, cookies))
            return results
        else:
            return None
        


if __name__ == "__main__":

    while True:
        if not ("09:20:00" <= datetime.now().strftime("%H:%M:%S") <= "15:35:00"):
            if datetime.now().strftime("%H:%M:%S") == "00:00:00":
                try:#as python interpreter will try to run this query multiple times as it will reach here multiple times in a sec 
                    conn = sqlite3.connect("stockOptionChainData.db")
                    cursor = conn.cursor()

                    # Get a list of all table names in the database
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()

                    # Iterate through each table and drop it
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                        print(f"Table '{table_name}' deleted.")

                    # Commit the changes and close the connection
                    conn.commit()
                    conn.close()
                except:
                    pass
            continue


        start = time.time()

        connection = sqlite3.connect("stockOptionChainData.db")
        cursor = connection.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS initialUnderlyingValues (
            symbol TEXT,
            underlyingValue REAL
        )
    ''')

        connection.commit()
        connection.close()

        with open("symbols.txt") as f:
            data = f.readlines()

        num_cores = multiprocessing.cpu_count()
        symbols = [i.strip('\n') for i in data]
        no_of_batches = 4*num_cores
        avg_elements = len(symbols) // no_of_batches
        remainder = len(symbols) % no_of_batches

        symbol_batches = [symbols[i * avg_elements + min(i, remainder):(i + 1) * avg_elements + min(i + 1, remainder)] for i in range(no_of_batches)]
            


        # Use the number of CPU cores as the number of processes
        with multiprocessing.Pool(processes=no_of_batches) as pool:
            # Use the pool to map the worker function onto the tasks
            results = pool.map(call_batch, symbol_batches)

        flattened_list = [item for sublist in results for item in sublist]
        print(f"\nPassed: {flattened_list.count(True)}\nFailed: {flattened_list.count(False)}")
        end = time.time()

        time.sleep(120-(end-start))


