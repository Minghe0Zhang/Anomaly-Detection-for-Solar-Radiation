import numpy as np
from predictor import event_predictor
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def demo_accuracy(filename):
    # num_loc   = 9
    data         = loadmat(f'For_Prediction/{filename}.mat')
    print(data.keys())

    beta_base_LS = data['beta_LSbaseline'][:,0]
    beta_int_LS  = data['beta_LSinter']
    beta_base_ML = data['beta_MLbaseline'][:,0]
    beta_int_ML  = data['beta_MLinter']
    K            = data['K'][0,0]                            # number of locations
    test_data    = data['test_data']
    print(test_data.shape)
    print(K)
    print(beta_int_ML.shape)

    test_data    = test_data.reshape(365,K)  # shape [365,K]
    # test_data    = np.load("For_Prediction/solar_mat.npy")
    d            = data['d']                                 # memory depth

    model = event_predictor(beta_base_LS, beta_int_LS, test_data)
    prob_events           = model.prob_predict()
    d_precision, d_recall, d_F1   = model.dynmamic_accuracy_metric()
    precision, recall, F1         = model.accuracy_metric(threshold=0.5)
    print(f"for LS method, {precision},{recall},{F1}")
    print(f"for LS method and dynmaic threshold, {d_precision},{d_recall},{d_F1}")


    # model2 = event_predictor(beta_base_ML, beta_int_ML, test_data)
    # prob_events = model2.prob_predict()
    # precision, recall, F1 = model2.accuracy_metric(threshold=0.5)
    # print(f"for ML method, {precision},{recall},{F1}")

    num_threshold = 100
    threshold     = np.linspace(0,1,num_threshold)
    precisions    = np.zeros(num_threshold)
    recalls       = np.zeros(num_threshold)
    F1s           = np.zeros(num_threshold)
    for i in range(len(threshold)):
        precisions[i], recalls[i], F1s[i] = model.accuracy_metric(threshold=threshold[i])


    fig = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(threshold, precisions, label='precision')
    plt.plot(threshold, recalls, label='recall')
    plt.plot(threshold, F1s, label='F1 score')

    # plt.vlines(0.5,0,1, linestyles ="dotted", colors='r')

    plt.legend(loc=1, prop={'size': 18})
    plt.xlabel("threshold",fontsize=24)
    plt.ylabel("metrics",fontsize=24)
    plt.title("Prediction Accuracy VS threshold",fontsize=24)
    plt.savefig(f"figs/{filename}_metricLS.pdf")


def demo_prob_events(filename):
    """
    plot figures for true events and predicted events 
    """
    data         = loadmat(f'For_Prediction/{filename}.mat')
    beta_base_LS = data['beta_LSbaseline'][:,0]
    beta_int_LS  = data['beta_LSinter']
    beta_base_ML = data['beta_MLbaseline'][:,0]
    beta_int_ML  = data['beta_MLinter']
    K            = data['K'][0,0]                            # number of locations
    test_data    = data['test_data']
    test_data    = test_data.reshape(365,K)  # shape [365,K]
    d            = data['d']                                 # memory depth

    model        = event_predictor(beta_base_LS, beta_int_LS, test_data)
    prob_events  = model.prob_predict()
    endpoint     = 365
    fig          = plt.figure(figsize=(20,5))
    x            = range(10+1,endpoint+1)
    d_th         = model.dynamic_th(prob_events,d2=10)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(x, prob_events[:endpoint,4], 'r*', label=r'Predicted Probability')
    plt.plot(x, test_data[10:endpoint+10,4],'k.', label=r'Ground Truth')
    plt.plot(x, d_th[:endpoint,4],'b.', label=r'Dynamic Threshold')
    plt.legend(loc=1, prop={'size': 18})
    plt.xlabel(r'Time',fontsize=24)
    plt.ylabel(r'Probability',fontsize=24)
    plt.title(r'Predicted Probability VS Ground Truth',fontsize=24)
    plt.savefig(f"figs/{filename}_predictLS.pdf")


    




if __name__ == "__main__":
    # locs = ['para_Fremont','para_Milpitas','para_Mountain View','para_North San Jose',
    #         'para_Palo Alto','para_Redwood City', 'para_San Mateo', 'para_Santa Clara', 
    #         'para_Sunnyvale', 'para_10_cities']
    locs   = ['para_10_cities']

    for loc in locs:
        demo_accuracy(loc)
        demo_prob_events(loc)

    # demo_prob_events(locs[0])