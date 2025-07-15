import pandas as pd
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from haversine import haversine, Unit
import numpy as np

def preprocess_data(df, test_size=0.2, random_state=1):
    df = clean_basic_columns(df)
    df['dist_to_cent'] = df.apply(compute_distance_to_center, axis=1)
    df = merge_sentiment_features(df)
    df = encode_neighbourhood(df)
    df = encode_room_type(df)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    impute_price(train_df, test_df)
    filtered_amenities = get_filtered_amenities(train_df)
    train_df = add_encoded_amenities(train_df, filtered_amenities)
    test_df = add_encoded_amenities(test_df, filtered_amenities)
    
    return train_df, test_df

def get_feature_groups(df):
    feature_groups = {
        "base": ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'dist_to_cent', 'minimum_nights', 'maximum_nights', 'host_is_superhost', 'host_total_listings_count', 'availability_eoy', 'host_listings_count', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'number_of_reviews_ly', 'estimated_occupancy_l365d', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'latitude', 'longitude', 'avg_sentiment', 'std_sentiment', 'min_sentiment', 'max_sentiment', 'instant_bookable'],
        "neighborhood": [col for col in df.columns if col.startswith('nh_')],
        "room_type": [col for col in df.columns if col.startswith('rt_')],
        "amenities": [col for col in df.columns if col.startswith('am_')],
    }
    return feature_groups

def clean_basic_columns(df):
    df = df.copy()
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})
    df['amenities'] = df['amenities'].apply(json.loads)
    return df

def compute_distance_to_center(row):
    city_center = (25.049589097552495, 121.5136645234154) # Taipei City Mall
    coords = (row['latitude'], row['longitude'])
    return haversine(city_center, coords, unit=Unit.KILOMETERS)

def merge_sentiment_features(df, sentiment_path="data/sentiment_features.csv"):
    sentiments = pd.read_csv(sentiment_path)
    df = pd.merge(df, sentiments, left_on='id', right_on='listing_id', how='left')
    
    df.drop(columns=['listing_id', 'sentiment_scores'], inplace=True, errors='ignore')
    return df

def encode_neighbourhood(df):
    nh_matrix = pd.get_dummies(df['neighbourhood_cleansed'], prefix='nh')
    df = pd.concat([df, nh_matrix], axis=1)
    return df

def encode_room_type(df):
    rt_matrix = pd.get_dummies(df['room_type'], prefix='rt')
    df = pd.concat([df, rt_matrix], axis=1)
    return df

def impute_price(train_df, test_df, column='price'):
    median = train_df[column].median()
    train_df[column] = train_df[column].fillna(median)
    test_df[column] = test_df[column].fillna(median)
    return train_df, test_df

def get_filtered_amenities(df_train, lower_threshold=500, upper_threshold=5500):
    all_amenities = [amenity for sublist in df_train['amenities'] for amenity in sublist]
    amenity_counts = Counter(all_amenities)

    filtered = {
        amenity
        for amenity, count in amenity_counts.items()
        if lower_threshold <= count <= upper_threshold
    }
    return filtered

def encode_amenities(row_amenities, filtered_amenities):
    return {
        f'am_{amenity}': int(amenity in row_amenities)
        for amenity in filtered_amenities
    }

def add_encoded_amenities(df, filtered_amenities):
    am_matrix = df['amenities'].apply(lambda x: encode_amenities(x, filtered_amenities)).apply(pd.Series)
    df = pd.concat([df, am_matrix], axis=1)
    return df