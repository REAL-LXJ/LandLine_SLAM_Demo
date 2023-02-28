# encoding:UTF-8 
import matplotlib.pyplot as plt


def _get_xy_data(sourceFile):
    x = []
    y = []
    with open(sourceFile, 'r') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            x.append(float(line[0]))
            y.append(float(line[1]))   
    return x, y

def _calculate_index(sourceFile):
    keypoint = ["FAST", "ShiTomas", "SIFT"]
    descriptor = ["BRIEF", "SIFT", "BELID"]
    average_time = 0
    sequence_start = 2
    sequence_end = 214
    frame_idx = 0
    run_time = 0
    sum_run_time = 0
    exp_name_num = sourceFile[-5:-4]
    if int(exp_name_num) == 1:
        feature_point = keypoint[0] + " + " + descriptor[0]
    elif int(exp_name_num) == 2:
        feature_point = keypoint[1] + " + " + descriptor[0]
    elif int(exp_name_num) == 3:
        feature_point = keypoint[2] + " + " + descriptor[1]
    elif int(exp_name_num) == 4:
        feature_point = keypoint[0] + " + " + descriptor[2]
    elif int(exp_name_num) == 5:
        feature_point = keypoint[1] + " + " + descriptor[2]

    source =  open(sourceFile, 'r') 

    for i in range(sequence_start, sequence_end + 1):
        sourceData = source.readline().strip('\n')
        sourceArray = sourceData.split(' ')
        frame_idx = sourceArray[0]
        run_time = sourceArray[1]
        sum_run_time += float(run_time)
    average_time = sum_run_time / (sequence_end - sequence_start + 1)
    print("feature_point: {}, average_time: {}".format(feature_point, average_time))
        

if __name__ == "__main__":
    exp1_file = "/home/lxj/桌面/lane_line_sfm/data/exp1.txt" # FAST + BRIEF
    exp2_file = "/home/lxj/桌面/lane_line_sfm/data/exp2.txt" # ShiTomas + BRIEF
    exp3_file = "/home/lxj/桌面/lane_line_sfm/data/exp3.txt" # SIFT + SIFT
    exp4_file = "/home/lxj/桌面/lane_line_sfm/data/exp4.txt" # FAST + BELID
    exp5_file = "/home/lxj/桌面/lane_line_sfm/data/exp5.txt" # ShiTomas + BELID
    depth = "/home/lxj/桌面/lane_line_sfm/data/frame_depth.txt"
    #_calculate_index(exp5_file)
    x, y = _get_xy_data(depth)
    #plt.xlabel("frame_idx")
    #plt.ylabel("run_time")
    #plt.title("different feature points")
    plt.xlabel("map_point_idx")
    plt.ylabel("depth")
    plt.title("map_point_depth")
    plt.plot(x, y)
    plt.show()