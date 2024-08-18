import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import osmnx as ox
import networkx as nx
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from streamlit_folium import folium_static

# Streamlit app title
st.title("Store Location Analysis and Route Optimization")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    dtf = pd.read_csv(uploaded_file)

    # Input City
    city = st.text_input("Enter the city name:", "London")
    dtf = dtf[dtf["City"] == city][["City", "Street Address", "Latitude", "Longitude"]].reset_index(drop=True)
    dtf = dtf.reset_index().rename(columns={"index": "id", "Latitude": "y", "Longitude": "x"})

    st.write(f"Total locations found: {len(dtf)}")
    st.write(dtf.head(3))

    data = dtf.copy()
    data["color"] = ''
    data.loc[data['id'] == 0, 'color'] = 'red'
    data.loc[data['id'] != 0, 'color'] = 'black'
    start = data[data["id"] == 0][["y", "x"]].values[0]

    # Display Folium Map
    map = folium.Map(location=start, tiles="cartodbpositron", zoom_start=12)
    data.apply(lambda row: 
        folium.CircleMarker(
            location=[row["y"], row["x"]], 
            color=row["color"], fill=True, radius=5).add_to(map), axis=1)

    folium_static(map)

    # Create the graph for routing
    G = ox.graph_from_point(start, dist=10000, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    start_node = ox.distance.nearest_nodes(G, start[1], start[0],method='kdtree')
    dtf["node"] = dtf[["y", "x"]].apply(lambda x: ox.distance.nearest_nodes(G, x[1], x[0]), axis=1)
    dtf = dtf.drop_duplicates("node", keep='first')

    def f(a, b):
        try:
            d = nx.shortest_path_length(G, source=a, target=b, method='dijkstra', weight='travel_time')
        except:
            d = np.nan
        return d

    distance_matrix = np.asarray([[f(a, b) for b in dtf["node"].tolist()] for a in dtf["node"].tolist()])
    distance_matrix = pd.DataFrame(distance_matrix, columns=dtf["node"].values, index=dtf["node"].values)

    # Display the distance matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(distance_matrix, vmin=0, vmax=1, cbar=False, ax=ax)
    st.pyplot(fig)

    # Routing model setup
    drivers = 1
    lst_nodes = dtf["node"].tolist()
    manager = pywrapcp.RoutingIndexManager(len(lst_nodes), drivers, lst_nodes.index(start_node))
    model = pywrapcp.RoutingModel(manager)

    def get_distance(from_index, to_index):
        return distance_matrix.iloc[from_index, to_index]

    distance = model.RegisterTransitCallback(get_distance)
    model.SetArcCostEvaluatorOfAllVehicles(distance)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = model.SolveWithParameters(parameters)

    index = model.Start(0)
    route_idx, route_distance = [], 0

    while not model.IsEnd(index):
        route_idx.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(model.NextVar(index))
        try:
            route_distance += get_distance(previous_index, index)
        except:
            route_distance += model.GetArcCostForVehicle(from_index=previous_index, to_index=index, vehicle=0)

    st.write(f"Route index: {route_idx}")
    st.write(f'Total distance: {round(route_distance / 1000, 2)} km')
    st.write(f'Nodes visited: {len(route_idx)}')

    lst_route = [lst_nodes[i] for i in route_idx]

    def get_path_between_nodes(lst_route):
        lst_paths = []
        for i in range(len(lst_route)):
            try:
                a, b = lst_route[i], lst_route[i + 1]
            except:
                break
            try:
                path = nx.shortest_path(G, source=a, target=b, method='dijkstra', weight='travel_time')
                if len(path) > 1:
                    lst_paths.append(path)
            except:
                continue
        return lst_paths

    lst_paths = get_path_between_nodes(lst_route)

    for path in lst_paths:
        ox.plot_route_folium(G, route=path, route_map=map, color="blue", weight=1)

    folium_static(map)

    def df_animation_multiple_path(G, lst_paths, parallel=True):
        df = pd.DataFrame()
        for path in lst_paths:
            lst_start, lst_end = [], []
            start_x, start_y = [], []
            end_x, end_y = [], []
            lst_length, lst_time = [], []

            for a, b in zip(path[:-1], path[1:]):
                lst_start.append(a)
                lst_end.append(b)
                lst_length.append(round(G.edges[(a, b, 0)]['length']))
                lst_time.append(round(G.edges[(a, b, 0)]['travel_time']))
                start_x.append(G.nodes[a]['x'])
                start_y.append(G.nodes[a]['y'])
                end_x.append(G.nodes[b]['x'])
                end_y.append(G.nodes[b]['y'])

            tmp = pd.DataFrame(list(zip(lst_start, lst_end, start_x, start_y, end_x, end_y, lst_length, lst_time)), 
                               columns=["start", "end", "start_x", "start_y", "end_x", "end_y", "length", "travel_time"])
            df = pd.concat([df, tmp], ignore_index=(not parallel))

        df = df.reset_index().rename(columns={"index": "id"})
        return df

    df = pd.DataFrame()
    tmp = df_animation_multiple_path(G, lst_paths, parallel=False)
    df = pd.concat([df, tmp], axis=0)
    first_node, last_node = lst_paths[0][0], lst_paths[-1][-1]
    df_start = df[df["start"] == first_node]
    df_end = df[df["end"] == last_node]

    fig = px.scatter_mapbox(data_frame=df, lon="start_x", lat="start_y", zoom=15, width=900, height=700, animation_frame="id", mapbox_style="carto-positron")
    fig.data[0].marker = {"size": 12}

    fig.add_trace(px.scatter_mapbox(data_frame=dtf, lon="x", lat="y").data[0])
    fig.data[1].marker = {"size": 10, "color": "black"}

    fig.add_trace(px.scatter_mapbox(data_frame=df_start, lon="start_x", lat="start_y").data[0])
    fig.data[2].marker = {"size": 15, "color": "red"}

    fig.add_trace(px.scatter_mapbox(data_frame=df_end, lon="start_x", lat="start_y").data[0])
    fig.data[3].marker = {"size": 15, "color": "green"}

    fig.add_trace(px.line_mapbox(data_frame=df, lon="start_x", lat="start_y").data[0])

    st.plotly_chart(fig)
