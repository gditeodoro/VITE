from training import *
from utils import *

if __name__=="__main__":

    max_depth = 4
    test_size=0.2
    max_features=None
    n_estimators=100

    x_tr, y_tr, x_ts, y_ts,data,scaler = readData("data/Cleveland.csv",test_size=test_size)

    model = trainModel(x_tr, y_tr,n_estimators=n_estimators,max_depth=max_depth,max_features=None)

    print("Evaluation on the training set")
    train_acc_RF,train_cm_RF = evaluateModel(model, x_tr, y_tr)

    print("Evaluation on the test set")
    test_acc_RF,test_cm_RF = evaluateModel(model, x_ts, y_ts)


    visualizationRF = VisualizationRF(model,data,max_depth,n_estimators) # inizializing the class of the visualization toolkit


    df_depth,df_perc = visualizationRF.stats_per_depth() #evaluating the number of features used at each depth and the

    visualizationRF.heatmap_RF_featuredepth(df_perc, 'figure', show=True)

    df_nodes_perc,tree_dict,df_nodes = visualizationRF.df_nodes_prc()

    visualizationRF.heatmap_nodes(df_nodes_perc, 'figure/nodes',tree_dict)
    visualizationRF.tree_heatmap("tree_heatmap",df_nodes_perc,'figure')

