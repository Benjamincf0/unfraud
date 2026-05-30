import pandas as pd
import numpy as np

def transform_dataset_for_training(input_filepath, output_filepath):
    print("Loading original dataset...")
    df_orig = pd.read_csv(input_filepath)
    
    # Initialize the new DataFrame
    df_new = pd.DataFrame()
    
    # 1. Base Mappings
    df_new['transaction_id'] = df_orig['trans_num']
    
    # Convert datetime to ISO format (with 'T')
    df_new['timestamp'] = pd.to_datetime(df_orig['trans_date_trans_time']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Convert credit card number to a generic card_id (e.g., "card_" + last 4 digits)
    df_new['card_id'] = 'card_' + df_orig['cc_num'].astype(str).str[-4:]
    
    df_new['amount'] = df_orig['amt']
    
    # Clean merchant name (Original data pre-pends 'fraud_' to the merchant string)
    df_new['merchant_name'] = df_orig['merchant'].str.replace('^fraud_', '', regex=True)
    df_new['merchant_category'] = df_orig['category']
    
    # 2. Derive new fields required by the target structure
    # If category contains 'net' or 'online', it's an online transaction, else in_person
    df_new['channel'] = np.where(
        df_orig['category'].str.contains('net', case=False, na=False), 
        'online', 
        'in_person'
    )
    
    # Default countries to 'US' based on original dataset location samples
    df_new['cardholder_country'] = 'US'
    df_new['merchant_country'] = 'US'
    
    # Simulate device_id and ip_address for online transactions (leave blank for in-person)
    # Note: We generate a pseudo-random device ID based on the index to keep it consistent
    df_new['device_id'] = np.where(
        df_new['channel'] == 'online', 
        'dev_' + df_orig.index.astype(str).str.zfill(8), 
        np.nan
    )
    
    # Assign a dummy IP address for online transactions, null for in-person
    df_new['ip_address'] = np.where(
        df_new['channel'] == 'online', 
        '192.168.1.' + (df_orig.index % 255).astype(str), 
        np.nan
    )
    
    # 3. EXTRA FIELDS FOR TRAINING
    # Target Variable
    df_new['is_fraud'] = df_orig['is_fraud']
    
    # Feature: User Age at time of transaction
    trans_year = pd.to_datetime(df_orig['trans_date_trans_time']).dt.year
    dob_year = pd.to_datetime(df_orig['dob']).dt.year
    df_new['user_age'] = trans_year - dob_year
    
    # Feature: Distance to merchant (using rough Euclidean distance in degrees for simplicity)
    # Note: For production, use the Haversine formula for exact kilometers/miles.
    df_new['distance_to_merchant'] = np.sqrt(
        (df_orig['lat'] - df_orig['merch_lat'])**2 + 
        (df_orig['long'] - df_orig['merch_long'])**2
    )
    
    # Feature: City Population
    df_new['city_pop'] = df_orig['city_pop']

    # 4. Save to CSV
    # Ensure column order starts with the required evaluation format, followed by ML features
    ordered_columns = [
        'transaction_id', 'timestamp', 'card_id', 'amount', 'merchant_name', 
        'merchant_category', 'channel', 'cardholder_country', 'merchant_country', 
        'device_id', 'ip_address', 
        'user_age', 'distance_to_merchant', 'city_pop', 'is_fraud'
    ]
    
    df_new = df_new[ordered_columns]
    df_new.to_csv(output_filepath, index=False)
    print(f"Data successfully transformed and saved to {output_filepath}")

# Usage Example:
# transform_dataset_for_training('original_fraud_data.csv', 'training_ready_data.csv')
if __name__ == "__main__":
    transform_dataset_for_training('fraudTest_unclean.csv', 'fraudTest.csv')
    transform_dataset_for_training('fraudTrain_unclean.csv', 'fraudTrain.csv')
