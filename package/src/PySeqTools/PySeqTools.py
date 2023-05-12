import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from mpl_toolkits.basemap import Basemap as Basemap

__version__ = "0.1.5"

def sim_func(a,b):
    """
    ## Description:
    
    The default similarity function to be passed in `seq_align_two_vars.propogate_matrix_global_two_vars` and `seq_align_two_vars.propogate_matrix_local_two_vars` functions. 
    
    ## Args:
    
    `a`: The first variable
        
    `b`: The second variable
    
    ## Returns:
    
    **1** if two varaibles are the same
    
    **-1** if two varaibles are not the same
    
    ## Example:
    
    ```python
    sim_func("a", "b")
    ```
    
    """
    if a == b:
        return 1
    else:
        return -1
    
def distance(val):
    """
    ## Description:
    
    The default function to convert the score calculated by `seq_align_two_vars.propogate_matrix_global_two_vars` or `seq_align_two_vars.propogate_matrix_local_two_vars` to non-negative value in order for distance, cluster, and later analysis. In those functions, if two varaibles are close to each other, they will assign a positive score. If two varaibles are different from each other, they will assign a very negative score. So if the value > 0, we will set the distance between two observations to 0. If the value <= 0, we will take the negative value as their distance.
    
    ## Args:
    
    `val`: The score need to be converted to non-negative value
    
    ## Returns:
    
    **0** if `val` > 0
    
    **-val** if `val` <= 0
    
    ## Example:
    
    ```python
    >>> distance(2)
    >>> 0
    >>> distance(-5)
    >>> 5
    ```
    
    """
    if val > 0:
        return 0
    else:
        return -val
    
# def pretty_print(matrix):
#     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))
#     print('\n')
    
    
# examples:
#citation networks

# To create data:
# 1. use make_sample
# 2. read in datafile
# 3. input networks, generate random walk data

# return: Dataframe with each row is an observation
# length > 3
def make_sample(length, category, size, var = 2, prob = None, distribution = np.random.uniform, var_len=False, seed = None, normalized = True):
    """
    ## Description:
    
    This function allows users to create randomly generated dataset for analysis. User need to pass in the length of each observation, the category for each observation, the number of observations. User can pass addition information like number of variable in each observation, like time, distance, the probability distribution for the category, the distribution for variable, and whether the length of each observation is constant or not. More details explained in the Args section.
    
    ## Args:
    `length`: The maximum length of each observation, required > 3
    
    `category`: A list of category in the dataset
    
    `size`: The number of observations
    
    `var`: default 2, the number of varaible in each observation. Default is set to 2 varaibles meaning time and distance
    
    `prob`: default None, a list of probabilty sum up to 1 for the category list you pass in. Default None meaning each category have the same probabilty to be chosen
    
    `distribution`: default `numpy.random.uniform`, the numpy probabilty distribution function to be used to create random value for the varaibale in each observation. Other possible value like `numpy.random.normal`, `numpy.random.exponential` and so on
    
    `var_len`: default False, the length of each observation if fixed or not. If True, then the length of each observation is a random integer from 3 to `length`. If False, then the length of each observation is `length`
    
    `seed`: default None, the seed for numpy random generator for reproduction of experiment
    
    `normalized`: default True, normalized all the varaible in the dataframe using `normalize` function
    
    ## Returns:
    
    A pandas DataFrame object with columns [category, var1, var2, ...,] and with rows representing each observation.
     
    """
    np.random.seed(seed)
    sample = []
    leng = length
    for k in range(size):
        if var_len == True:
            leng = np.random.randint(3, length)
        cate_lst = []
        lst = [cate_lst]
        for i in range(var):
            x = distribution(size = leng + 1)
            lst.append(x)

        for i in range(leng):
            if prob != None:
                cur = category[np.random.choice(np.arange(len(category)), p=prob)]
            else:
                cur = category[np.random.randint(0, len(category))]
            cate_lst.append(cur)
        sample.append(lst)
    if normalized == True:
        return normalize(pd.DataFrame(sample, columns= ["category"] + [str(i+1) for i in range(var)]))
    else:
        return pd.DataFrame(sample, columns= ["category"] + [str(i+1) for i in range(var)])

def read_data(filename, column_name, category_index = 0, normalized = True):
    """
    ## Description:
    
    This function read the data and process the data into the correct format that can be passed to further analysis. The file currently allowed is csv, json, excel, parquet.gzip.
    
    ## Args:
    
    `filename`: The name of the file
    
    `column_name`: A list contain The name of the column need to be read in
    
    `category_index`: default 0, the index of the category column in the `column_name` list
    
    `normalized`: default True, normalized all the varaible in the dataframe using `normalize` function
    
    ## Returns:
    
    A pandas DataFrame object with columns [category, var1, var2, ...,] and with rows representing each observation.
     
    """
    if "parquet.gzip" in filename:
        df = pd.read_parquet(filename, columns = column_name)
    elif "csv" in filename:
        df = pd.read_csv(filename, names= column_name)
    elif "json" in filename:
        df = pd.read_json(filename, names= column_name)
    elif "excel" in filename:
        df = pd.read_excel(filename, names= column_name)
    else:
        df = pd.read_data(filename, names= column_name)
    dct = {}
    count = 1
    for i in range(len(column_name)):
        if  i == category_index:
            dct[column_name[i]] = "category"
        else:
            dct[column_name[i]] = count
            count += 1
    df.rename(columns = dct, inplace=True)
    df = df[['category'] + [ col for col in df.columns if col != 'category']]
    if normalized == True:
        return normalize(df)
    else:
        return df

    
def geo_read_data(filename, column_name, loc_index = 0, category_index = 1, normalized = True):
    """
    ## Description:
    
    For visualizing geolocation of sequence movement on the map. This function read the data and process the data into the correct format that can be passed to further analysis. The file currently allowed is csv, json, excel, parquet.gzip.
    
    ## Args:
    
    `filename`: The name of the file
    
    `column_name`: A list contain The name of the column need to be read in
    
    `loc_index`: default 0, the index of the location column in the `column_name` list
    
    `category_index`: default 1, the index of the category column in the `column_name` list
    
    `normalized`: default True, normalized all the varaible in the dataframe using `normalize` function
    
    ## Returns:
    
    A pandas DataFrame object with columns [category, var1, var2, ...,] and with rows representing each observation.
     
    """
    if "parquet.gzip" in filename:
        df = pd.read_parquet(filename, columns = column_name)
    elif "csv" in filename:
        df = pd.read_csv(filename, names= column_name)
    elif "json" in filename:
        df = pd.read_json(filename, names= column_name)
    elif "excel" in filename:
        df = pd.read_excel(filename, names= column_name)
    else:
        df = pd.read_data(filename, names= column_name)
    dct = {}
    count = 1
    for i in range(len(column_name)):
        if  i == category_index:
            dct[column_name[i]] = "category"
        elif i == loc_index:
            dct[column_name[i]] = "location"
        else:
            dct[column_name[i]] = count
            count += 1
    df.rename(columns = dct, inplace=True)
    df = df[['location', 'category'] + [ col for col in df.columns if col != 'category' and col != 'location']]
    if normalized == True:
        return normalize(df)
    else:
        return df

#normalized 
def normalize(df):
    """
    ## Description:
    
    This function is a helper function to normalize the processed dataframe from `make_sample`, `read_data`, or `random_walk`. It normalized all the columns except category column to make all the value between 0 and 1. It find the maximum value and minimum value from each column, then for each entry x, perform (x - min) / (max - min).
    
    ## Args:
    
    `df`: the dataframe processed by `make_sample`, `read_data`, or `random_walk`
    
    ## Returns:
    
    A pandas DataFrame object with columns [category, var1, var2, ...,] and with rows representing each observation. Each value in every column have been normalized between 0 and 1.
     
    """
    for i in df.columns:
        if i != "category" and i != "location":
            min_ = 100000000000000000
            max_ = 0
            for j, row in df.iterrows():
                for num in row[i]:
                    if num > max_:
                        max_ = num
                    if num < min_:
                        min_ = num
            lst = []
            for j, row in df.iterrows():
                tmp = []
                for num in row[i]:
                    tmp.append(round((num - min_) / (max_ - min_), 4))
                lst.append(tmp)
            
            df[i] = lst
    return df

#only work with random variable now

#weighted edge and node network
#logistic network
def random_walk(G, walk_len, count, var=2, distribution=np.random.uniform, distance_matrix = None, seed = None, normalized = True):
    """
    ## Description:
    
    This function requires a networkx network and the length of the walk, and the number of observations to create a bunch of random walk on the network user passed in. Then it randomly generate the variable data base on distribution function user passed in.
    
    ## Args:
    
    `G`: network passed in as networkx format
    
    `walk_len`: The maximum length of each random walk
    
    `count`: The number of observations
    
    `var`: default 2, the number of varaible in each observation. Default is set to 2 varaibles meaning time and distance
    
    `distribution`: default `numpy.random.uniform`, the numpy probabilty distribution function to be used to create random value for the varaibale in each observation. Other possible value like `numpy.random.normal`, `numpy.random.exponential` and so on
    
    `distance_matrix`: default None, randomly generate the distance matrix for the distance between every node in the network
    
    `seed`: default None, the seed for numpy random generator for reproduction of experiment
    
    `normalized`: default True, normalized all the varaible in the dataframe using `normalize` function
    
    ## Returns:
    
    A pandas DataFrame object with columns [category, var1, var2, ...,] and with rows representing each observation. Each value in every column have been normalized between 0 and 1.
     
    """
    np.random.seed(seed)
    if distance_matrix == None:
        distance_matrix = np.random.random((len(list(G.nodes)), len(list(G.nodes))))
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    walks = []
    for i in range(count):
        cur_node = np.random.choice(list(G.nodes))
        lst = [[cur_node]]
        x = []
        for j in range(1, walk_len):
            neigh = list(G.neighbors(cur_node))
            if neigh:
                neighbor = np.random.choice(neigh)   # choose one random neighbor
                lst[0].append(neighbor)   # add it to walk
                x.append(distance_matrix[cur_node][neighbor])
                cur_node = neighbor   # save it to cur_node to continue from it next iteration
            else:
                break
        for j in range(var-1):
            y = distribution(size = len(lst[0]) - 1)
            lst.append(y)
        lst.append(x)
        walks.append(lst)
    if normalized == True:
        return normalize(pd.DataFrame(walks, columns= ["category"] + [str(i+1) for i in range(var)]))
    else:
        return pd.DataFrame(walks, columns= ["category"] + [str(i+1) for i in range(var)])

def get_cmap(n, name='hsv'):
    '''
    ## Description:
    
    helper function for the `draw`, use to generate color map based on index and colormap name.
    
    ## Args:
    
    `n`: number of colors wanted
    
    `name`: default hsv, must be a standard matplotlib colormap name
    
    ## Returns:
    
    a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    '''
    return plt.cm.get_cmap(name, n)

#TODO: edge length not thickness
def draw(df, sample, color = "hsv"):
    '''
    ## Description:
    
    Using networkx to visualize the observation with two varaibles representing time and distance. The node size is proportional to time and the edge thickness is proportional to distance. Every category is represented by a different color.
    
    ## Args:
    
    `df`: the dataframe processed by `make_sample`, `read_data`, or `random_walk`
    
    `sample`: a list of index of the observation wanted to visualize
    
    `color`: default hsv, can be either a standard matplotlib colormap name or a dict specifying category and corresponding color. e.g. {"A": "r", "B": "y", "C": "g", "D":"b"}
    
    ## Returns:
    
    None. Plot a networkx graph.

    '''
    # only work for var = 2, best suitable to show data inteprated as time and dis
    # color example: {"A": "r", "B": "y", "C": "g", "D":"b"} or cmap
    tmp = []
    for s in sample:
        for i in df.loc[s].iloc[0]:
            if i not in tmp:
                tmp.append(i)
    
    if isinstance(color, dict) == False:
        cmap = get_cmap(len(tmp)+1, name = color)
        color = {}
        for i in range(len(tmp)):
            color[tmp[i]] = cmap(i)
    
    for s in sample:
        G=nx.DiGraph()
        X = df.loc[s].iloc[0]
        X1 = df.loc[s][1]
        X2 = df.loc[s][2]

        plt.figure(1,figsize=(2.5 * len(X),2)) 

        node_size = []
        for i in range(len(X) - 1):
            node_size.append(1000 + 10000 * float(X1[i]))
        node_size.append(1000)

        node_color = []
        width = []
        sum_dis = 0
        for i in range(len(X)):
            if i == 0:
                G.add_node(i, pos = (0,1))
            elif i != len(X) - 1:
                diss = (np.sqrt(node_size[i]/31400)-np.sqrt(2000/31400) + np.sqrt(node_size[i-1]/31400)-np.sqrt(2000/31400))**2
                sum_dis = sum_dis + diss + 1
                G.add_node(i, pos = (sum_dis,1))
            else:
                G.add_node(i, pos = (sum_dis+1,1))
            node_color.append(color[X[i]])

        edge_labels = {}
        for i in range(len(X) - 1):
            G.add_edge(i, i+1, len= 1)
            edge_labels[(i, i+1)] = round(float(X2[i]),2)
            width.append(0.2+float(X2[i]) * 10)
        pos=nx.get_node_attributes(G,'pos')

        labeldict = {}

        for i in range(len(X) - 1):
            labeldict[i] = str(X[i]) + "\n" + str(round(float(X1[i]),2))
        labeldict[i+1] = X[i+1]

        nx.draw(G,pos, labels = labeldict, node_size = node_size, width = width, node_color = node_color, edge_color = 'b')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()
        
def geo_draw(df, sample, df_loc, use_map = False, map1 = None, lllon=-82,
        lllat=39,
        urlon=-74,
        urlat=43, color = "hsv"):
    '''
    ## Description:
    
    
    Using networkx to visualize the observation with two varaibles representing time and distance. The node size is proportional to time and the edge thickness is proportional to distance. Every category is represented by a different color. Also show the nodes on geolocally on map.
    
    ## Args:
    
    `df`: the dataframe processed by `make_sample`, `read_data`, or `random_walk`
    
    `sample`: a list of index of the observation wanted to visualize
    
    `df_loc`: the latitude and longtidue for all the location, stored in dataframe
    
    `use_map`: default False, you can pass your own map to plot on
    
    `map1`: your own map
    
    `lllon`: left lower longtitude
    
    `lllat`: left lower latitude
    
    `urlon`: upper right longtitude
    
    `urlat`: upper right latitude
    
    `color`: default hsv, can be either a standard matplotlib colormap name or a dict specifying category and corresponding color. e.g. {"A": "r", "B": "y", "C": "g", "D":"b"}
    
    ## Returns:
    
    None. Plot a networkx graph.

    '''
    # only work for var = 2, best suitable to show data inteprated as time and dis
    # color example: {"A": "r", "B": "y", "C": "g", "D":"b"} or cmap
    tmp = []
    for s in sample:
        for i in df.loc[s].iloc[1]:
            if i not in tmp:
                tmp.append(i)
    
    if isinstance(color, dict) == False:
        cmap = get_cmap(len(tmp)+1, name = color)
        color = {}
        for i in range(len(tmp)):
            color[tmp[i]] = cmap(i)
    
    for s in sample:
        G=nx.DiGraph()
        
        loca = df.loc[s].iloc[0]
        X = df.loc[s].iloc[1]
        X1 = df.loc[s][1]
        X1.append(0)
        X2 = df.loc[s][2]

        plt.figure(1,figsize=(2.5 * len(X),2)) 

        node_size = []
        for i in range(len(X) - 1):
            node_size.append(1000 + 10000 * float(X1[i]))
        node_size.append(1000)

        node_color = []
        width = []
        sum_dis = 0
        for i in range(len(X)):
            if i == 0:
                G.add_node(i, pos = (0,1))
            elif i != len(X) - 1:
                diss = (np.sqrt(node_size[i]/31400)-np.sqrt(2000/31400) + np.sqrt(node_size[i-1]/31400)-np.sqrt(2000/31400))**2
                sum_dis = sum_dis + diss + 1
                G.add_node(i, pos = (sum_dis,1))
            else:
                G.add_node(i, pos = (sum_dis+1,1))
            node_color.append(color[X[i]])

        edge_labels = {}
        for i in range(len(X) - 1):
            G.add_edge(i, i+1, len= 1)
            edge_labels[(i, i+1)] = round(float(X2[i]),2)
            width.append(0.2+float(X2[i]) * 10)
        pos=nx.get_node_attributes(G,'pos')

        labeldict = {}

        for i in range(len(X) - 1):
            labeldict[i] = str(X[i]) + "\n" + str(round(float(X1[i]),2))
        labeldict[i+1] = X[i+1]

        nx.draw(G,pos, labels = labeldict, node_size = node_size, width = width, node_color = node_color, edge_color = 'b')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()
        
        if use_map == False:
            m = Basemap(
            projection='merc',
            llcrnrlon=lllon,
            llcrnrlat=lllat,
            urcrnrlon=urlon,
            urcrnrlat=urlat,
            lat_ts=0,
            resolution='i',
            suppress_ticks=True)

            plt.figure(figsize=(12,12))

            pos={}
            node_color = []
            node_size = []

            G=nx.DiGraph()

            for i in range(len(X)):
                G.add_node(loca[i])
                lat = df_loc.loc[loca[i]]["lat"]
                lon = df_loc.loc[loca[i]]["long"]
                mx,my = m(lon, lat)
                pos[loca[i]] = (mx,my)

            for i in range(len(X) - 1):
                G.add_edge(loca[i], loca[i+1])

            for x in list(pos.keys()):
                node_color.append(color[X[np.where(np.array(loca) == x)[0][0]]])
                node_size.append(400 + 2000 * float(X1[np.where(np.array(loca) == x)[0][0]]))

            nx.draw_networkx(G,pos,node_color=node_color, node_size = node_size)

            # Now draw the map
            m.drawcountries()
            m.drawstates()
            m.drawrivers()
            m.bluemarble()
            plt.show()
        else:
            BBox = ((lllon,urlon,lllat,urlat))
            fig, ax = plt.subplots(figsize = (12,12))
            ax.imshow(map1,zorder=0, extent = BBox)

            pos={}
            node_color = []
            node_size = []

            G=nx.DiGraph()

            for i in range(len(X)):
                G.add_node(loca[i])
                lat = df_loc.loc[loca[i]]["lat"]
                lon = df_loc.loc[loca[i]]["long"]
                pos[loca[i]] = (lon, lat)

            for i in range(len(X) - 1):
                G.add_edge(loca[i], loca[i+1])

            for x in list(pos.keys()):
                node_color.append(color[X[np.where(np.array(loca) == x)[0][0]]])
                node_size.append(400 + 2000 * float(X1[np.where(np.array(loca) == x)[0][0]]))

            nx.draw_networkx(G,pos,node_color=node_color, node_size = node_size)

            plt.show()

# def com_draw(df, lst, color = "hsv"):
#     '''
#     ## Description:
    
#     Functions allow drawing multiple observations at the same time using `draw` function.
    
#     ## Args:
    
#     `df`: dataframe processed by `normalize`
    
#     `lst`: the list of the index of the observations user wanted to visualize
    
#     `color`: default hsv, can be either a standard matplotlib colormap name or a dict specifying category and corresponding color. e.g. {"A": "r", "B": "y", "C": "g", "D":"b"}
    
#     ## Returns:
    
#     None. Plot a group of networkx graphs.

#     '''
#     for x in lst:
#         draw(df, x, color = color)
    
def propogate_matrix_global_two_vars(X, Y, X1, Y1, X2, Y2, ratio, gap_score, align_score, proportion):
    '''
    ## Description:
    
    The function calculates the similarity of two observations with two varaibles using global sequence alignment method.
    
    ## Args:
    
    `X`: first observation category column
    
    `Y`: second observation category column
    
    `X1`: first observation 1st variable column
    
    `Y1`: second observation 1st variable column
    
    `X2`: first observation 2nd variable column
    
    `Y2`: second observation 2nd varaible column
    
    `ratio`: value between 0 and 1, meaning the weight of the first variable in the similarity calculation. If ratio is 1, meaning we only use first variable. If ratio is 0, meaning we only use second variable. If ratio is in between, meaning the X/Y1 * ratio + X/Y2 * (1-ratio).
    
    `gap_score`: Constant value, meaning how much gap penalty give for sequence alignment
    
    `align_score`: The similarity function like `sim_func`, you can create your own similarity function and pass in
    
    `proportion`: The weight of the variable comparsion in the similarity score calculate, usual value 5, 10, 20
    
    ## Returns:
    
    Integer that represents the similarity value between these two observations
    
    ## Example:
    
    ```python
    propogate_matrix_global_two_vars(df.loc[i]["category"], df.loc[j]["category"], df.loc[i]["1"], df.loc[j]["1"], df.loc[i]["2"], df.loc[j]["2"], 0.5, 1, sim_func, 2)
    ```
    '''
    X1 = np.insert(X1, 0, 0)
    X2 = np.insert(X2, 0, 0)
    Y1 = np.insert(Y1, 0, 0)
    Y2 = np.insert(Y2, 0, 0)
    
    value_matrix = [[0 for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    source_matrix = [[[0, 0] for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    source_gap_score = [[0 for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    need_constant_gap = [[0 for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    
    need_constant_gap[0][0] = 1
    
    for j in range(len(Y)+1):
        for i in range(len(X)+1):
            if j == 0 and i == 0:
                continue
            # For the global approach, the first row and column involves the time penalties
            # of adding in a gap of a given length to the beginning of either sequence.
            elif j == 0:
                above_score = update_gap_two_vars(i - 1, j, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "above")
                above_value = value_matrix[i-1][j] - gap_score * need_constant_gap[i-1][j] + source_gap_score[i-1][j] * proportion - above_score[0] * proportion
                    
                value_matrix[i][j] = above_value
                source_matrix[i][j] = [i - 1,j]
                source_gap_score[i][j] = above_score[0]
                need_constant_gap[i][j] = 0
            elif i == 0:
                left_score = update_gap_two_vars(i, j - 1, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "left")
                left_value = value_matrix[i][j-1] - gap_score * need_constant_gap[i][j-1] + source_gap_score[i][j-1] * proportion - left_score[0] * proportion
                    
                value_matrix[i][j] = left_value
                source_matrix[i][j] = [i,j-1]
                source_gap_score[i][j] = left_score[0]
                need_constant_gap[i][j] = 0
            else:
                score = align_score(X[i-1], Y[j-1])
                
                diag_score = update_gap_two_vars(i-1, j-1, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "diag")
                diag_value = value_matrix[i-1][j-1] + score + source_gap_score[i-1][j-1] * proportion- diag_score[0] * proportion
                    
                left_score = update_gap_two_vars(i, j-1, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "left")
                left_value = value_matrix[i][j-1] - gap_score * need_constant_gap[i][j-1] + source_gap_score[i][j-1] * proportion - left_score[0] * proportion
                
                above_score = update_gap_two_vars(i-1, j, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "above")
                above_value = value_matrix[i-1][j] - gap_score * need_constant_gap[i-1][j] + source_gap_score[i-1][j] * proportion - above_score[0] * proportion
                
                max_score = max(diag_value, left_value, above_value)
                value_matrix[i][j] = max_score
                if diag_value == max_score:
                    source_matrix[i][j] = [i -1, j-1]
                    source_gap_score[i][j] = 0
                    need_constant_gap[i][j] = 1
                elif left_value == max_score:
                    source_matrix[i][j] = [i,j-1]
                    if left_score[1] or above_score[1]:
                        source_gap_score[i][j] = 0
                        need_constant_gap[i][j] = 1
                    else:
                        source_gap_score[i][j] = left_score[0]
                        need_constant_gap[i][j] = 0
                else:
                    source_matrix[i][j] = [i -1,j]
                    if left_score[1] or above_score[1]:
                        source_gap_score[i][j] = 0
                        need_constant_gap[i][j] = 1
                    else:
                        source_gap_score[i][j] = above_score[0]
                        need_constant_gap[i][j] = 0
    # pretty_print(value_matrix)
    return value_matrix[len(X)][len(Y)]

def propogate_matrix_local_two_vars(X, Y, X1, Y1, X2, Y2, ratio, gap_score, align_score, proportion):
    '''
    ## Description:
    
    The function calculates the similarity of two observations with two varaibles using local sequence alignment method.
    
    ## Args:
    
    `X`: first observation category column
    
    `Y`: second observation category column
    
    `X1`: first observation 1st variable column
    
    `Y1`: second observation 1st variable column
    
    `X2`: first observation 2nd variable column
    
    `Y2`: second observation 2nd varaible column
    
    `ratio`: value between 0 and 1, meaning the weight of the first variable in the similarity calculation. If ratio is 1, meaning we only use first variable. If ratio is 0, meaning we only use second variable. If ratio is in between, meaning the X/Y1 * ratio + X/Y2 * (1-ratio).
    
    `gap_score`: Constant value, meaning how much gap penalty give for sequence alignment
    
    `align_score`: The similarity function like `sim_func`, you can create your own similarity function and pass in
    
    `proportion`: The weight of the variable comparsion in the similarity score calculate, usual value 5, 10, 20
    
    ## Returns:
    
    Integer that represents the similarity value between these two observations
    
    ## Example: 
    ```python
    propogate_matrix_local_two_vars(df.loc[i]["category"], df.loc[j]["category"], df.loc[i]["1"], df.loc[j]["1"], df.loc[i]["2"], df.loc[j]["2"], 0.5, 1, sim_func, 2)
    ```
    '''
    X1 = np.insert(X1, 0, 0)
    X2 = np.insert(X2, 0, 0)
    Y1 = np.insert(Y1, 0, 0)
    Y2 = np.insert(Y2, 0, 0)
    
    value_matrix = [[0 for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    source_matrix = [[[0, 0] for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    source_gap_score = [[0 for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    need_constant_gap = [[0 for x in range(len(Y) + 1)] for x in range(len(X) + 1)]
    
    need_constant_gap[0][0] = 1
    
    for j in range(1, len(Y)+1):
        for i in range(1, len(X)+1):
            score = align_score(X[i-1], Y[j-1])

            diag_score = update_gap_two_vars(i-1, j-1, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "diag")
            diag_value = value_matrix[i-1][j-1] + score + source_gap_score[i-1][j-1] * proportion- diag_score[0] * proportion

            left_score = update_gap_two_vars(i, j-1, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "left")
            left_value = value_matrix[i][j-1] - gap_score * need_constant_gap[i][j-1] + source_gap_score[i][j-1] * proportion - left_score[0] * proportion

            above_score = update_gap_two_vars(i-1, j, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, "above")
            above_value = value_matrix[i-1][j] - gap_score * need_constant_gap[i-1][j] + source_gap_score[i-1][j] * proportion - above_score[0] * proportion

            max_score = max(diag_value, left_value, above_value, 0)
            value_matrix[i][j] = max_score
            if diag_value == max_score or max_score == 0:
                source_matrix[i][j] = [i-1, j-1]
                source_gap_score[i][j] = 0
                need_constant_gap[i][j] = 1
            elif left_value == max_score:
                source_matrix[i][j] = [i,j-1]
                if left_score[1] or above_score[1]:
                    source_gap_score[i][j] = 0
                    need_constant_gap[i][j] = 1
                else:
                    source_gap_score[i][j] = left_score[0]
                    need_constant_gap[i][j] = 0
            else:
                source_matrix[i][j] = [i -1,j]
                if left_score[1] or above_score[1]:
                    source_gap_score[i][j] = 0
                    need_constant_gap[i][j] = 1
                else:
                    source_gap_score[i][j] = above_score[0]
                    need_constant_gap[i][j] = 0
    # pretty_print(value_matrix)
    return value_matrix[len(X)][len(Y)]

# ratio 0-1, X/Y1 * ratio + X/Y2 * (1-ratio)
def update_gap_two_vars(row_index, column_index, source_matrix, X1, Y1, X2, Y2, ratio, source_gap_score, direction):
    '''
    ## Description:
    
    Helper function for `propogate_matrix_global_two_vars` and `propagate_matrix_local_two_vars` to calculate the similarity of two observations with two varaibles. This function use to update the gap penalty between two variables in the sequence.
    
    ## Args:
    
    `row_index`: current row in the score matrix to be filled in
    
    `column_index`: current column in the score matrix to be filled in
    
    `score_matrix`: the score matrix to store the score for every row and every column
    
    `X1`: first observation 1st variable column
    
    `Y1`: second observation 1st variable column
    
    `X2`: first observation 2nd variable column
    
    `Y2`: second observation 2nd varaible column
    
    `ratio`: value between 0 and 1, meaning the weight of the first variable in the similarity calculation. If ratio is 1, meaning we only use first variable. If ratio is 0, meaning we only use second variable. If ratio is in between, meaning the X/Y1 * ratio + X/Y2 * (1-ratio).
    
    `source_gap_score`: Constant value, meaning how much gap penalty give for sequence alignment
    
    `direction`: "left", "above", "diag", meaning which is the way the current score calculated based on: left entry, above entry, or diagonal entry.
    
    ## Returns:
    
    Integer that used as penalty score in the `propogate_matrix_global_two_vars` and `propagate_matrix_local_two_vars` method
    '''
    # This means we are dealing with a value alongside the edge.
    if row_index == 0 or column_index == 0:
        return [abs(sum(X1[0:row_index+1]) - sum(Y1[0:column_index+1])) * ratio + abs(sum(X2[0:row_index+1]) - sum(Y2[0:column_index+1])) * (1-ratio), 0]
    # This means this value came from our diagonal direction.
    elif source_matrix[row_index][column_index][0] < row_index and source_matrix[row_index][column_index][1] < column_index:
        if direction == "left":
            return [Y1[column_index] * ratio + Y2[column_index] * (1-ratio), 0]
        elif direction == "above":
            return [X1[row_index] * ratio + X2[row_index] * (1-ratio), 0]
        else:
            return [abs(X1[row_index] - Y1[column_index]) * ratio + abs(X2[row_index] - Y2[column_index]) * (1-ratio), 0]
    # In this case, this value came from a downward movement, meaning an extended gap in the y-direction.
    elif source_matrix[row_index][column_index][0] < row_index:
        # This means that our best choice is a 'zigzag' movement.  So, we need to have the algorithm
        # reset the gap score, since we are now going to deal with a gap in the other sequence.
        if direction == "left":
            return [abs(source_gap_score[row_index][column_index] - Y1[column_index] * ratio - Y2[column_index] * (1-ratio)), 1]
        elif direction == "above":
            return [source_gap_score[row_index][column_index] + X1[row_index] * ratio + X2[row_index] * (1-ratio), 0]
        else:
            return [abs(source_gap_score[row_index][column_index] + (X1[row_index] - Y1[column_index]) * ratio + (X2[row_index] - Y2[column_index]) * (1-ratio)), 0]
    # In this case, this value came from a rightward movement, meaning an extended gap in the x-direction.
    elif source_matrix[row_index][column_index][1] < column_index:
        if direction == "left":
            return [source_gap_score[row_index][column_index] + Y1[column_index] * ratio + Y2[column_index] * (1-ratio), 0]
        # This means that our best choice is a 'zigzag' movement.  So, we need to have the algorithm
        # reset the gap score, since we are now going to deal with a gap in the other sequence.
        elif direction == "above":
            return [abs(source_gap_score[row_index][column_index] - X1[row_index] * ratio - X2[row_index] * (1-ratio)), 1]
        else:
            return [abs(source_gap_score[row_index][column_index] + (Y1[column_index ] - X1[row_index])*ratio + (Y2[column_index] - X2[row_index]) * (1-ratio)), 0]
        
#analysis
def distance_metric(df, method= "global", treat_dis = True):
    '''
    ## Description:
    
    Calculate the distance between each observation in the dataframe, return a distance matrix, with each entry represent the distance between ith and jth entry calculated by `propogate_matrix_global_two_vars` or `propogate_matrix_local_two_vars`. This is a symmetric matrix.
    
    ## Args:
    
    `df`: the dataframe processed by `make_sample`, `read_data`, or `random_walk`
    
    `method`: default global, if global, we will use `propogate_matrix_global_two_vars` to calculate the similarity score, if local, we will use `propogate_matrix_local_two_vars` to calculate the similarity score
    
    `treat_dis`: default True, if True, we will treat the similarity score as distance using `distance`, if False, we will keep the similarity score as it is
    
    ## Returns:
    
    A distance matrix
    '''
    dist = np.zeros((len(df),len(df)))
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if method == "local":
                dis = propogate_matrix_local_vars(df.loc[i]["category"], df.loc[j]["category"], df.loc[i]["1"], df.loc[j]["1"], df.loc[i]["2"], df.loc[j]["2"], 0.5, 1, sim_func, 2)
            else:
                dis = propogate_matrix_global_two_vars(df.loc[i]["category"], df.loc[j]["category"], df.loc[i]["1"], df.loc[j]["1"], df.loc[i]["2"], df.loc[j]["2"], 0.5, 1, sim_func, 2)
            if treat_dis:
                dis = distance(dis)
            dist[i][j] = dis
            dist[j][i] = dis
    return dist

def neighbours(dis, target, thres):
    '''
    ## Description:
    
    This funciton use distance matrix to find neighbour of a specific target within a distance threshold.
    
    ## Args:
    
    `dis`: distance matrix returned by `distance_metric`
    
    `target`: the index of the observation to find it neighbours
    
    `thres`: the threshold distance for an observation to be considered as the neighbour
    
    ## Returns:
    
    A list contain all the index of the observations that has a distance less than the threshold
    '''
    count = 0
    lst = []
    for x in dis[target]:
        if x < thres & count != target:
            lst.append(count)
        count += 1
    return lst

def cluster(dis, distance = None, n_clusters = None):
    '''
    ## Description:
    
    Use sklearn Agglomerative Clustering to cluster based on distance matrix, given distance threshold or number of clusters.
    
    ## Args:
    
    `dis`: distance matrix returned by `distance_metric`
    
    `distance`: default None, distance threshold to be considered as one cluster
    
    `n_clusters`: default None, the number of clusters user wanted
    
    ## Returns:
    
    sklearn.AgglomerativeClustering object can be used in further function
    
    '''
    model = AgglomerativeClustering(metric="precomputed", distance_threshold=distance,n_clusters=n_clusters, linkage="average", compute_distances=True)
    model = model.fit(dis)
    return model

def plot_dendrogram(model, **kwargs):
    '''
    ## Description:
    
    Use the clustering model to plot dendrogram 
    
    ## Args:
    
    `model`: model returned by `cluster`
    
    `kwargs`: according to scipy.cluster.hierarchy.dendrogram kwargs
    
    ## Returns:
    
    dendrogram plot by the model
    
    '''
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def count_clusters(model):
    '''
    ## Description:
    
    Function to count how many elements in each cluster given the model
    
    ## Args:
    
    `model`: model returned by `cluster`
    
    ## Returns:
    
    Two lists contains the number of cluster and elements in each cluster
    
    '''
    labels = np.array(model.labels_)
    unique, counts = np.unique(labels, return_counts=True)
    return unique, counts