import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import folium_static

# Function to fetch GPS data from a CSV file
def fetch_gps_data():
    try:
        # Replace with the path to your CSV file
        gps_data = pd.read_csv('path_to_your_gps_data.csv')
        return gps_data
    except Exception as e:
        st.error(f"Error fetching GPS data: {e}")
        return None

# Function to create a graph from GPS data
def create_graph(gps_data):
    G = nx.Graph()
    for idx, row in gps_data.iterrows():
        G.add_node(row['id'], lat=row['lat'], lon=row['lon'])
        if idx > 0:
            G.add_edge(gps_data.iloc[idx-1]['id'], row['id'], weight=1)  # Simplified example
    return G

# Function to calculate travel time (assuming a constant speed)
def calculate_travel_time(path, gps_data, speed_kph=50):
    distance = 0
    for i in range(len(path) - 1):
        source = gps_data[gps_data['id'] == path[i]].iloc[0]
        target = gps_data[gps_data['id'] == path[i+1]].iloc[0]
        distance += haversine_distance(source['lat'], source['lon'], target['lat'], target['lon'])
    time = distance / speed_kph  # Time in hours
    return distance, time

# Haversine formula to calculate distance between two GPS points
def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to plot paths on a map
def plot_paths(gps_data, original_path, optimized_path):
    # Create a folium map centered at the midpoint of the path
    midpoint = len(gps_data) // 2
    m = folium.Map(location=[gps_data.iloc[midpoint]['lat'], gps_data.iloc[midpoint]['lon']], zoom_start=12)

    # Plot original path
    for i in range(len(original_path) - 1):
        source = gps_data[gps_data['id'] == original_path[i]].iloc[0]
        target = gps_data[gps_data['id'] == original_path[i+1]].iloc[0]
        folium.PolyLine([(source['lat'], source['lon']), (target['lat'], target['lon'])],
                        color='blue', weight=2.5, opacity=1).add_to(m)

    # Plot optimized path
    for i in range(len(optimized_path) - 1):
        source = gps_data[gps_data['id'] == optimized_path[i]].iloc[0]
        target = gps_data[gps_data['id'] == optimized_path[i+1]].iloc[0]
        folium.PolyLine([(source['lat'], source['lon']), (target['lat'], target['lon'])],
                        color='green', weight=2.5, opacity=1).add_to(m)

    return m

# Main Streamlit app
def main():
    st.title("Route Optimization and Mapping")

    # Fetch and display GPS data
    gps_data = fetch_gps_data()

    if gps_data is not None:
        st.write("GPS Data", gps_data.head())

        # Create a graph from GPS data
        graph = create_graph(gps_data)

        # Original path (assuming it’s just a straight line from start to end)
        original_path = list(gps_data['id'])

        # Optimize the route using Dijkstra's algorithm
        source = gps_data.iloc[0]['id']
        target = gps_data.iloc[-1]['id']
        optimized_path = nx.shortest_path(graph, source=source, target=target, weight='weight')

        # Calculate distances and travel times
        original_distance, original_time = calculate_travel_time(original_path, gps_data)
        optimized_distance, optimized_time = calculate_travel_time(optimized_path, gps_data)

        # Display calculated times
        st.write(f"Original Path: {original_distance:.2f} km, Estimated Time: {original_time:.2f} hours")
        st.write(f"Optimized Path: {optimized_distance:.2f} km, Estimated Time: {optimized_time:.2f} hours")

        # Plot and display paths on a map
        map_plot = plot_paths(gps_data, original_path, optimized_path)
        folium_static(map_plot)
    else:
        st.error("Failed to fetch GPS data.")

if __name__ == "__main__":
    main()
