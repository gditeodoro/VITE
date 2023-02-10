import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import graphviz
import os
import warnings
warnings.filterwarnings("ignore")

class VisualizationRF:

    def __init__(self,trained_model,data,max_depth,n_estimators):

        self.model = trained_model
        self.data = data
        self.max_depth = max_depth
        self.list_features = list(self.data.columns)[:-1]
        self.n_estimators = n_estimators

    def plot_model(self):
        '''
        :param model: trained classifier
        :param list_features: a list with the range of the number of the features e.g. [0,1,2,3,4] if I have 4 features
        :return: dict_tree, a dictionary with keys a string "tree_#tree" and as values dictionaries with the string of the
        number of the node as key and a list with [depth,feature,threshold] as values
        '''
        features = len(self.list_features)
        tree_dict = {}
        results = {
            'tree_num': list(range(len(self.model.estimators_))),
            **{f"{f}": [0] * len(self.model.estimators_) for f in range(features)},
            **{f"{f}_depth": [[]] * len(self.model.estimators_) for f in range(features)}
        }

        for i, estimator in enumerate(self.model.estimators_):
            tree_dict[f"tree_{i}"] = {}
            for j in range(len(estimator.tree_.children_left)):
                node, depth, feat = self.getDepth(estimator.tree_, j)
                tree_dict[f"tree_{i}"][node] = [depth, feat, estimator.tree_.threshold[node]]
                if feat != -2:
                    results[f"{feat}"][i] += 1
                    results[f"{feat}_depth"][i].append(depth)
                    results = pd.DataFrame.from_dict(results)

        return results, tree_dict



    def getDepth(self, tree_, node, root=0):
        '''
        :param tree_: the estimator tree of the forest
        :param node: the number relative to the considered node
        :param root: the number relative to the root node
        :return: the number of the node, the depth where the node is located, the feature used in that node
        '''

        depth = 0
        ancestor = node
        while ancestor != root:
            children = tree_.children_left if ancestor in tree_.children_left else tree_.children_right
            ancestor = np.where(children == ancestor)[0][0]
            depth += 1
        return node, depth, tree_.feature[node]

    def tot_depth_list(self, results):
        '''
        :param features_name:  list with the name of the features
        :param results: list with all the depth used in the tree
        :return: a list of all the depth that I can find in all the trees
        '''
        depths = [d for feature in list(range(len(self.list_features))) for depth_list in
                  results[f"{feature}_depth"].values for d in depth_list]
        return sorted(list(set(depths)))

    def depth_perc_DataFrame(self, tot_depth, results):
        '''
        :param tot_depth: list with the total number of depth of the trees, enumerated
        :param results: a dictionary with keys a string "tree_#tree" and as values dictionaries with the string of the
        number of the node as key and a list with [depth,feature,threshold] as values
        :param features_name: list with the name features
        :return: two dataframes: df_depth that is a df #features X #depth that say how many time each feature is used at each depth
        df_perc: same as df_depth but in percentage (frequencies)
        '''
        n_features = len(self.list_features)
        n_depths = len(tot_depth)
        depth = np.zeros((n_features, n_depths))

        feature_depth = [results[f"{f}_depth"].values for f in range(n_features)]
        depth = np.array([np.bincount(np.hstack(f_depth).astype(np.int64), minlength=n_depths) for f_depth in feature_depth])

        df_depth = pd.DataFrame(depth, columns=['depth_' + str(d) for d in tot_depth])
        df_perc = df_depth / df_depth.sum(axis=0)
        df_perc.rename(columns={'depth_' + str(ix): 'level_' + str(ix) for ix in tot_depth}, inplace=True)
        df_perc.rename(index={i: self.list_features[i] for i in range(n_features)}, inplace=True)

        return df_depth, df_perc


    def heatmap_RF_featuredepth(self,df_perc,directory='figure',show=None):
        '''
        :param df_perc: df #features X #depth that say how many time each feature is used at each depth, in percentage
        :param directory: directory where to save the heatmap
        :param show: if True, the heatmap is displayed
        :return: an heatmap to visualize the features that are more used, at each depth, in the whole forest
        '''
        # HEATMAP
        fig, ax = plt.subplots(figsize=(len(df_perc.index)/2,len(df_perc.index)/2))

        # Add title to the Heat map
        title = "RF Heatmap with % of features usage per level"

        # Set the font size and the distance of the title from the plot
        plt.title(title,fontsize=13)
        ttl = ax.title
        ttl.set_position([0.5,1.05])

        # Hide ticks for X & Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove the axes
        #sns.heatmap(df_perc,annot =np.array(df_perc.values*100,dtype='int'),fmt="",cmap="YlGnBu",linewidths=0.30,ax=ax)
        map = sns.color_palette("Blues", as_cmap=True)

        sns.heatmap(df_perc, annot=np.array(df_perc.values * 100, dtype='int'), fmt="", cmap=map, linewidths=0.30,
                    ax=ax)
        if show ==True:
            # Display the Heatmap
            plt.show()

        #fig.savefig(path+"heat_map.png")
        fig.savefig(os.path.join(os.getcwd(),directory,"heat_map.png"))
        plt.close()


    def node_same_name_(self,tree_dict):
        '''
        :param tree_dict: dictionary of dictionaries, with keys the name of the tree and values a dictionary with
        keys the integer of the node and value a list with [depth of the node,feature used in the node,threshold uded
        in that node]
        :param max_depth: depth max of the trees
        :return: tree_dict with the number that indicate the name of the nodes, unified in a way that node in the
        same position in different trees are named with the same integer
        '''

        n = [1]  # inizializing the list that indicates the number of nodes at each depth
        for iter in range(self.max_depth):
            n.append(n[-1] * 2)
        num_nodes = sum(
            n[:-1])  # number of nodes of the representation ( the last depth is not represented since there is
        # not a split in the last level of the tree)
        num_nodes_tot = sum(n)  # number of nodes, leaves included

        # loop to rename the nodes ell with the same name througout the tree
        for tree_ in list(tree_dict.keys()):
            jump_index = 0

            for node in list(range(num_nodes_tot)):
                app_list = []
                app_dict = {}
                if node in list(tree_dict[tree_].keys()) and tree_dict[tree_][node][1] == -2 and tree_dict[tree_][node][0] != self.max_depth:
                    jump = int(sum(n[tree_dict[tree_][node][0]+1:])/(n[tree_dict[tree_][node][0]]))
                    if node!=num_nodes_tot-3:
                        for nod in list(tree_dict[tree_].keys())[node+1-jump_index:]:
                            app = tree_dict[tree_][nod]
                            app_list.append(nod)
                            app_dict[nod+jump] = app
                        jump_index += jump
                    for nod in app_list:
                        del tree_dict[tree_][nod]
                    tree_dict[tree_] = {**tree_dict[tree_],**app_dict}
        return tree_dict

    def node_depth_dict(self):
        '''
        :param max_depth: macimum depth of each tree of the forest
        :return: a dictionary with keys a string that indicate the depth (e.g. 'd_0'=depth 0 ) and values a list of the
        integer of the nodes at that depth
        '''


        n = list(2 ** np.arange(self.max_depth + 1))

        n_under = [(sum(n[num + 1:]) / n[num]) for num in range(len(n))]
        dict_node_depth = {f'd_{i}': [] for i in range(self.max_depth + 1)}
        dict_node_depth['d_0'] = [0]
        for d in range(1,self.max_depth + 1):
            for el in dict_node_depth['d_' + str(d - 1)]:
                dict_node_depth['d_' + str(d)].extend([int(el + 1), int(el + 2 + n_under[d])])

        return dict_node_depth

    def heatmap_nodes(self,df_nodes_perc,directory,tree_dict):
        '''
        :param df_nodes_perc: dataframe that have as index the features and columns the nodes of the tree
        in the df is saved the number of time each feature is used in each node
        :param directory: directory where to save the images
        :param tree_dict: dictionary of dictionaries, with keys the name of the tree and values a dictionary with
        keys the integer of the node and value a list with [depth of the node,feature used in the node,threshold uded
        in that node]
        :param max_depth: depth max of the trees
        :param features_name: list with the name of the features
        :return: the images of the heatmap to save at each node
        '''

        n = [1]  # inizializing the list that indicates the number of nodes at each depth
        for iter in range(self.max_depth):
            n.append(n[-1] * 2)
        num_nodes_tot = sum(n)  # number of nodes, leaves included

        threshold_dict = {'node_'+str(n):{} for n in range(num_nodes_tot)} #inizializing the dictionary of thresholds
        for i in range(num_nodes_tot):
            threshold_dict['node_'+str(i)] = {f : [] for f in self.list_features}

        for tree_ in list(tree_dict.keys()):
            for node in range(num_nodes_tot):
                if node in tree_dict[tree_].keys() and tree_dict[tree_][node][2]!=-2 :
                    threshold_dict['node_'+str(node)][self.list_features[tree_dict[tree_][node][1]]].append(tree_dict[tree_][node][2])
                #else:
                    # threshold_dict['node_'+str(node)].append(None)
        df_threshold = pd.DataFrame(threshold_dict)

        node_name = list(df_nodes_perc.columns)

        for name in node_name:
            num_node = int(name.split('_')[1])
            # HEATMAP
            fig, ax = plt.subplots(figsize=(len(df_nodes_perc.index)/4,len(df_nodes_perc.index)/3.5))#,len(df_nodes_perc.index)*2 ))
            fig.set_tight_layout(True)
            # Add title to the Heat map
            title = "_%"
            # Set the font size and the distance of the title from the plot
            plt.title(title, fontsize=18)
            ttl = ax.title
            ttl.set_position([0.5, 1.05])
            # Hide ticks for X & Y axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(ylabel=None)


            df_torep = pd.DataFrame(df_nodes_perc[name]).T
            for f in self.list_features:
                if len(threshold_dict[name][f]) != 0:
                    inf = round((min(threshold_dict[name][f])),2)
                    sup = round((max(threshold_dict[name][f])),2)
                    if inf ==sup:
                        df_torep.rename(columns={f:f+' '+str(inf)
                                         }, inplace=True)
                    else:
                        df_torep.rename(columns={f: f + ' ' + '['+str(inf)+','+str(sup)+']'
                                                 }, inplace=True)
            df_torep.rename(index={name:''},inplace=True)
            # Remove the axes
            map = sns.color_palette("Blues", as_cmap=True)
            sns.heatmap(df_torep, annot=np.array(df_torep.values * 100, dtype='int'), fmt="", cmap=map, linewidths=0.30,
                        ax=ax)

            fig.savefig(os.path.join(os.getcwd(),directory,"heatmap_"+name+".png"))
            plt.close(fig)



    def tree_heatmap(self,filename,df_nodes_perc,directory):
        '''
        function to represent the tree with heatmap at each node
        :param filename: name of the pdf file where to save the file
        :param df_nodes_perc: dataframe with the percentage with which each feature is used in a given node
        :param max_depth: maximum depth of the tree in the random forest
        :param directory: directory where to save the images
        :return: a pdf file with the tree with heatmap
        '''
        f = graphviz.Digraph(filename= filename,format='png')
        df_support = df_nodes_perc.dropna(axis=1)
        names = list(df_support.columns)

        positions = ['' for i in range(len(names))]
        for name, position in zip(names, positions):
            f.node(name, position,
                   image=os.path.join(os.getcwd(),directory,'nodes','heatmap_'+name+'.png'),shape="plaintext")
        node_depth = {f"node_{nd}": int(key.split("_")[1]) for key, values in self.node_depth_dict().items() if
                      key != f"d_{self.max_depth}" for nd in values}


        for d in range(1,self.max_depth):
            n_p0 = [ k for k,v in node_depth.items() if v==d-1]
            n_p1 = [ k for k,v in node_depth.items() if v==d]
            for elem in n_p0:
                if len(n_p0)!=0:
                    for i in range(2):
                        f.edge(elem,n_p1[i])
                    n_p0 = n_p0[1:]
                    n_p1 = n_p1[2:]

        #f.render(directory=os.getcwd(), view=True)
        f.render(directory=os.path.join(os.getcwd(),directory), view=True).replace('\\', '/')


    def stats_per_depth(self):
        results,tree_dict = self.plot_model()
        tot_depth = self.tot_depth_list(results)
        df_depth, df_perc = self.depth_perc_DataFrame(tot_depth, results)

        return df_depth,df_perc

    def df_nodes_prc(self):
        '''
        :return: dataframe with the percentage of features at each node
        '''
        results, tree_dict = self.plot_model()
        tree_dict = self.node_same_name_(tree_dict)

        n = [1]
        for iter in range(self.max_depth):
            n.append(n[-1] * 2)

        num_nodes_tot = sum(n)  # number of nodes, leaves included
        nodes = {'node_' + str(d): [] for d in range(num_nodes_tot)}
        dict_node_depth = self.node_depth_dict()

        for f,_ in enumerate(self.list_features):
            f_count = np.zeros(num_nodes_tot)
            for tree_, nodes_ in tree_dict.items():
                for node in list(nodes_.keys()):
                    if tree_dict[tree_][node][1] == f:  # node<num_nodes and
                        f_count[node] += 1
            for nn in range(num_nodes_tot):
                nodes['node_' + str(nn)].append(f_count[nn])


        df_nodes = pd.DataFrame.from_dict(nodes)  # columns are depth and rows are features

        df_nodes_perc = df_nodes.div(df_nodes.sum(axis=0), axis=1)
        df_nodes_perc.rename(index={i: self.list_features[i] for i in range(len(self.list_features))}, inplace=True)
        df_nodes_perc.drop(['node_' + str(i) for i in dict_node_depth['d_' + str(self.max_depth)]], axis=1,
                           inplace=True)


        return df_nodes_perc,tree_dict,df_nodes



