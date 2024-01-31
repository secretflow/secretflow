import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# cosine similarity
def get_cos_sim(v1, v2):
  num = float(np.dot(v1,v2))
  denom = np.linalg.norm(v1) * np.linalg.norm(v2)
  # return num / denom if denom != 0 else 0
  return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# claculate detection score
def detection_score(sl_model_path, fsha_model_path):
    plt.cla()
    poly_n = 2
    dir_path = '.'
    marker_size = 3
    similarity = np.load(sl_model_path + '.npy')[:, :4000]

    plt.rcParams['axes.unicode_minus'] = False 
    n_len = similarity.shape[1]
    x = np.arange(n_len)
    x2 = np.arange(50, n_len, 50)

    z1 = np.polyfit(x, similarity[0], poly_n)
    p1 = np.poly1d(z1)
    pp1 = p1(x2)
    plt.plot(x2, similarity[0][50::50], color='dodgerblue', linestyle='-', marker='o', label='different category data', markersize=marker_size, alpha=0.9)

    z2 = np.polyfit(x, similarity[1], poly_n)
    p2 = np.poly1d(z2)
    pp2 = p2(x2)
    plt.plot(x2, similarity[1][50::50], color='red', linestyle='-', marker='o', label='same category data', markersize=marker_size, alpha=0.9)

    same_variance_min = similarity[1][50::50] - similarity[3][50::50]
    same_variance_max = similarity[1][50::50] + similarity[3][50::50]
    diff_variance_min = similarity[0][50::50] - similarity[2][50::50]
    diff_variance_max = similarity[0][50::50] + similarity[2][50::50]
    iou_in = np.where(np.maximum(same_variance_min, diff_variance_min)> np.minimum(diff_variance_max, same_variance_max),
                      np.zeros_like(same_variance_min),np.minimum(diff_variance_max, same_variance_max)-np.minimum(same_variance_min, diff_variance_min))
    iou_range = np.maximum(diff_variance_max, same_variance_max)-np.minimum(same_variance_min, diff_variance_min)
    iou_score = iou_in/iou_range

    bool_mar = (np.maximum(same_variance_min, diff_variance_min) > np.minimum(diff_variance_max, same_variance_max))
    in_range_max = np.minimum(diff_variance_max, same_variance_max)
    in_range_max[bool_mar] = np.nan
    in_range_min = np.maximum(same_variance_min, diff_variance_min)
    in_range_min[bool_mar] = np.nan
    in_range_index = x2

    plt.fill_between(x2, same_variance_min, same_variance_max,
                     color='lightcoral', alpha=0.7) #229 / 256, 204 / 256, 249 / 256
    plt.fill_between(x2, diff_variance_min, diff_variance_max,
                     color='lightskyblue', alpha=0.7) #(204 / 256, 236 / 256, 223 / 256)
    plt.fill_between(in_range_index, in_range_min, in_range_max,
                     color='blueviolet', alpha=0.8)
    plt.ylim((0.0, 1.0))
    # plt.legend()
    plt.xlabel('steps')  
    plt.ylabel('gradient cosine similarity')
    plt.xticks(np.arange(50, 3333, 400))
    plt.savefig('honest_training.pdf')
    plt.show()
    plt.clf()

    gap_score = (similarity[1, :] - similarity[0, :]) / (similarity[0, :] + similarity[1, :])

    err_score = []
    for i in range(50, n_len, 50):
        x_current = np.arange(i)
        z_current = np.polyfit(x_current, similarity[0][:i], poly_n)
        p1_current = np.poly1d(z_current)
        pp1_current = p1_current(x_current)
        err = np.sqrt(np.mean((similarity[0][:i]-pp1_current)**2))
        err_score.append(err)
    err_score = np.array(err_score)


    err_diff_score = []
    for i in range(50, n_len, 50):
        x_current = np.arange(i)
        z_current = np.polyfit(x_current, similarity[1][:i], poly_n)
        p1_current = np.poly1d(z_current)
        pp1_current = p1_current(x_current)
        err = np.sqrt(np.mean((similarity[1][:i] - pp1_current) ** 2))
        err_diff_score.append(err)
    err_diff_score = np.array(err_diff_score)

    similarity = np.load(fsha_model_path + '.npy')[:,:4000]
    plt.rcParams['axes.unicode_minus'] = False 
    n_len = similarity.shape[1]
    x = np.arange(n_len)
    x2 = np.arange(50, n_len, 50)
    z1 = np.polyfit(x, similarity[0], poly_n)
    p1 = np.poly1d(z1)
    pp1 = p1(x2)
    plt.plot(x2, similarity[0][50::50], color='dodgerblue', linestyle='-', marker='o', label='different category data', markersize=marker_size, alpha=0.9)

    z2 = np.polyfit(x, similarity[1], poly_n)
    p2 = np.poly1d(z2)
    pp2 = p2(x2)
    plt.plot(x2, similarity[1][50::50], color='red', linestyle='-', marker='o', label='same category data', markersize=marker_size, alpha=0.9 )

    same_variance_min = similarity[1][50::50] - similarity[3][50::50]
    same_variance_max = similarity[1][50::50] + similarity[3][50::50]
    diff_variance_min = similarity[0][50::50] - similarity[2][50::50]
    diff_variance_max = similarity[0][50::50] + similarity[2][50::50]

    bool_mar = (np.maximum(same_variance_min, diff_variance_min) > np.minimum(diff_variance_max, same_variance_max))
    in_range_max = np.minimum(diff_variance_max, same_variance_max)
    in_range_max[bool_mar] = np.nan
    in_range_min = np.maximum(same_variance_min, diff_variance_min)
    in_range_min[bool_mar] = np.nan
    in_range_index = x2

    FSHA_iou_in = np.where(
        np.maximum(same_variance_min, diff_variance_min) > np.minimum(diff_variance_max, same_variance_max),
        np.zeros_like(same_variance_min),
        np.minimum(diff_variance_max, same_variance_max) - np.maximum(same_variance_min, diff_variance_min))
    FSHA_iou_range = np.maximum(diff_variance_max, same_variance_max) - np.minimum(same_variance_min, diff_variance_min)
    FSHA_iou_score = FSHA_iou_in / FSHA_iou_range

    plt.fill_between(x2, same_variance_min, same_variance_max,
                     color='lightcoral', alpha=0.7)  # 229 / 256, 204 / 256, 249 / 256
    plt.fill_between(x2, diff_variance_min, diff_variance_max,
                     color='lightskyblue', alpha=0.7)  # (204 / 256, 236 / 256, 223 / 256)
    plt.fill_between(in_range_index, in_range_min, in_range_max,
                     color='blueviolet', alpha=0.8)

    # plt.legend()
    plt.xlabel('steps') 
    plt.ylabel('gradient cosine similarity')
    plt.ylim((0.0, 1.))
    plt.xticks(np.arange(50, 3333, 400))
    plt.savefig('fsha_training.pdf')
    plt.show()
    plt.clf()

    FSHA_err_score = []
    for i in range(50, n_len, 50):
        x_current = np.arange(i)
        z_current = np.polyfit(x_current, similarity[0][:i], poly_n)
        p1_current = np.poly1d(z_current)
        pp1_current = p1_current(x_current)
        err = np.sqrt(np.mean((similarity[0][:i] - pp1_current) ** 2))
        FSHA_err_score.append(err)
    FSHA_err_score = np.array(FSHA_err_score)

    FSHA_err_diff_score = []
    for i in range(50, n_len, 50):
        x_current = np.arange(i)
        z_current = np.polyfit(x_current, similarity[1][:i], poly_n)
        p1_current = np.poly1d(z_current)
        pp1_current = p1_current(x_current)
        err = np.sqrt(np.mean((similarity[1][:i] - pp1_current) ** 2))
        FSHA_err_diff_score.append(err)
    FSHA_err_diff_score = np.array(FSHA_err_diff_score)

    x_err = np.arange(50, n_len, 50)
    plt.plot(x_err, (err_diff_score+err_score)/2, color='tomato', linestyle='-', marker='',
             label='Honest ', markersize=marker_size)
    plt.plot(x_err, (FSHA_err_score+FSHA_err_diff_score)/2., color='dodgerblue', linestyle='-', marker='',
             label='FSHA', markersize=marker_size)
    plt.ylim((0.0, 0.07))
    plt.legend()
    plt.xlabel('step') 
    plt.ylabel('fitting error')
    plt.xticks(np.arange(50, 4000, 400))
    plt.savefig('error_score.pdf')
    plt.show()
    plt.clf()

    x_gap = np.arange(50, n_len, 50)
    plt.plot(x_gap, gap_score[50::50], color='tomato', linestyle='-', marker='',
             label='Honest', markersize=marker_size)
    gap_score_fsha = (similarity[1, :] - similarity[0, :]) / (similarity[0, :] + similarity[1, :])
    plt.plot(x_gap, gap_score_fsha[50::50], color='dodgerblue', linestyle='-', marker='',
             label='FSHA', markersize=marker_size)
    plt.ylim((0.0, 0.3))
    plt.legend()
    plt.xlabel('step') 
    plt.ylabel('gap score')
    plt.xticks(np.arange(50, 4000, 400))
    plt.savefig('gap_score.pdf')
    plt.show()
    plt.clf()

    x_iou = np.arange(50, n_len, 50)
    plt.plot(x_iou, iou_score, color='tomato', linestyle='-', marker='',
             label='Honest', markersize=marker_size)
    plt.plot(x_iou, FSHA_iou_score, color='dodgerblue', linestyle='-', marker='',
             label='FSHA', markersize=marker_size)
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.xlabel('step')  # x轴的名字
    plt.ylabel('overlapping ratio')
    plt.xticks(np.arange(50, 4000, 400))
    plt.savefig('overlapping_iou_score.pdf')
    plt.show()
    plt.clf()

    # hyperparameters for detection score
    epsilo = 1/(np.e*np.e)
    la = 7
    la_g = 0.6
    la_ds = 5
    la_iou = 1
    eposilo_iou = 1/np.e

    # FSHA detection score calculation
    FSHA_modified_err = (FSHA_err_score+FSHA_err_diff_score)/2*la
    FSHA_m_err = -np.log( FSHA_modified_err + epsilo)
    FSHA_m_iou = -np.log(FSHA_iou_score * la_iou + eposilo_iou)
    FSHA_m_gap = gap_score_fsha[50::50]/(np.abs(gap_score_fsha[50::50])**la_g)
    FSHA_final_error = FSHA_m_gap * FSHA_m_err * FSHA_m_iou -0.7
    FSHA_final_error = sigmoid(la_ds*FSHA_final_error)

    # Honest training detection score calculation
    modified_err = (err_score + err_diff_score) / 2 * la
    m_err = -np.log(modified_err + epsilo)
    m_iou = -np.log(iou_score*la_iou + eposilo_iou)
    m_gap = gap_score[50::50]/(np.abs(gap_score[50::50])**la_g)
    final_error = m_gap * m_err * m_iou -0.7
    final_error = sigmoid(la_ds *final_error)

    x_err = np.arange(50, n_len, 50)
    plt.plot(x_err, FSHA_final_error, color='dodgerblue', linestyle='-', marker='',
             label='FSHA', markersize=marker_size)
    plt.plot(x_err, final_error, color='red', linestyle='-', marker='',
             label='Honest', markersize=marker_size)
    plt.axhline(y=0.5, xmin=0, xmax=1, color='orange', linestyle='-.', marker='',
             label='detection threshold', markersize=marker_size)
    plt.ylim((-0.05, 1.05))
    plt.xlabel('steps')
    plt.ylabel('detection score')
    plt.xticks(np.arange(50, 3333, 400))
    plt.savefig('detection_score.pdf')
    plt.show()
    plt.clf()

    return [FSHA_final_error, final_error]