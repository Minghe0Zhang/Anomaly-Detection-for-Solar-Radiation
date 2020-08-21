import numpy as np
import geopandas
import branca
import folium
from shapely.geometry import Polygon
from scipy.io import loadmat
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib as mpl
from IPython.display import display




def mat_read(file):
    x     = loadmat(file)
    trMLH = x['trMLH']
    aff   = trMLH['aff'][0][0]
    base  = trMLH['new'][0][0]
    print(f"aff: {aff.shape}")
    print(f"base: {base.shape}")
    # print(aff[:,:,0,0,1])
    return aff[:,:,0,0,1],base

def spatio_plot(grid,adj,base):
    K = grid**2
    G = nx.DiGraph()
    for i in range(K):
        loc = (i//grid,i%grid)
        G.add_node(i,pos=loc,weight=base[i][0])


    for i in range(K):
        for j in range(K):
            if(adj[i,j]!=0):
                G.add_edge(i, j, weight=adj[i,j])
    pos=nx.get_node_attributes(G,'pos')
    labels = nx.get_node_attributes(G,'weight')
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())


    nx.draw(G, pos, node_size=[(v*200)**2 for v in base],labels=labels, node_color='b', edgelist=edges, edge_color=weights, 
            width=10.0, edge_cmap=plt.cm.Blues)
    # pc = mpl.collections.PatchCollection(base, cmap=plt.cm.Blues)
    # plt.colorbar()    
    plt.show()

def plot_on_map(grid=3):

    """
    Plot some points on the maps, the dots contains an list of [longtitude,lantitutde]
    """
    loc = [33.749-0.038, -84.388-0.038]
    m = folium.Map(
        location=loc
    )
    for i in range(grid):
        for j in range(grid):
            folium.CircleMarker(
                location=[loc[0]+0.038*i, loc[1]+0.038*j],
                radius=20,
                fill=True, # Set fill to True
                fill_color="blue"
                # popup='Mt. Hood Meadows'
                # icon=folium.Icon(icon='cloud')
            ).add_to(m)
    m.save("map.html")
    # display(m)
    



if __name__ == "__main__":
    plot_on_map(grid=3)
    # adj, base = mat_read("map_vis.mat")
    # base =  np.around(base, decimals=3)
    # spatio_plot(grid=3, adj=adj, base=base)