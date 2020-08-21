import numpy as np
import matplotlib.pyplot as plt
import raw2abnormal

def test_gen_rawdata():
    scale = 3
    output_name = "solar2017_3by3"
    raw2abnormal.API_extraction(output_name,scale)
    # raw2abnormal.newAPI()

def test_daily_abnormal():
    winsize = 20
    delta   = 0.0005
    abnormal = raw2abnormal.ToAbnormalByDay("../solar2018_3by3.csv",winsize,delta)
    # print(abnormal)
    raw2abnormal.ToMatrix_daily(abnormal,grid=3)
    # return abnormal


def test_daily_multistate():
    winsize = 20
    delta   = 0.0005
    abnormal,abnormal_neg = raw2abnormal.ToAbnormalByDay("../solar2018_3by3.csv",winsize,delta,single=False)
    raw2abnormal.ToMatrix_daily_multi(abnormal, abnormal_neg, grid=3)

def test_hourly_abnormal():
    winsize = 31
    delta   = 0.0005
    abnormal = raw2abnormal.ToAbnormalByHour("../solar2018_3by3.csv",winsize,delta)
    # print(abnormal)
    raw2abnormal.ToMatrix_hourly(abnormal,grid=3)

def raster_plot():
    winsize = 20
    delta   = 0.001
    # abnormal = raw2abnormal.ToAbnormalByDay("../solar2018_3by3.csv",winsize,delta)
    abnormal,abnormal_neg = raw2abnormal.ToAbnormalByDay("../solar2018_3by3.csv",winsize,delta,single=False)

    fig = plt.figure()
    ax = plt.subplot(111)


    mat = raw2abnormal.ToMatrix_daily(abnormal,grid=3)
    mat = mat.reshape(365,9)
    for i in range(9):
        for j in range(100):
            if mat[j,i]!= 0:
                ax.plot(j,i+1,'ro')
    ax.plot(j,i+1,'ro',label = "Postive abnormal events")

    mat_neg = raw2abnormal.ToMatrix_daily(abnormal_neg,grid=3)
    mat_neg = mat_neg.reshape(365,9)
    for i in range(9):
        for j in range(100):
            if mat_neg[j,i]!= 0:
                ax.plot(j,i+1,'ko')
    ax.plot(j,i+1,'ko',label = "Negative abnormal events")


    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    plt.title("Raster plot for two types of abnormal events in solar system")
    plt.xlabel("time")
    plt.ylabel("location ID (9 in total)")
    plt.savefig("raster_medium_delta_multiple.pdf")



if __name__ == "__main__":
    test_gen_rawdata()
    # test_daily_abnormal()
    # test_daily_multistate()
    # test_hourly_abnormal()
    # raster_plot()
    
