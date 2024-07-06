
# ==========PERCEPTRON===========================
# T = 90000
# theta for Perceptron is 3.640699999921921, 4.24459999999955
# theta_0 for Perceptron is -7.0

# T = 100000
# theta for Perceptron is 4.119399999913389, 2.7907999999996242
# theta_0 for Perceptron is -7.0


# T = 110000
# theta for Perceptron is 2.6029999999046716, 3.7967999999994753
# theta_0 for Perceptron is -7.0

# T = 120000
# theta for Perceptron is 2.853799999896003, 5.570599999999502
# theta_0 for Perceptron is -8.0


# ==========AVG PERCEPTRON===========================Converges

# T = 110000
# theta for Average Perceptron is 3.850249907215709, 3.8882350022497674
# theta_0 for Average Perceptron is -7.0544635

# T = 120000
# theta for Average Perceptron is 3.8499083092061244, 3.8875930961294882
# theta_0 for Average Perceptron is -7.0534995

# T = 130000
# theta for Average Perceptron is 3.849521929550608, 3.8874154850813034
# theta_0 for Average Perceptron is -7.0531185



# ==========Pegasos===========================converges
# T = 130000
# theta for Pegasos is 0.6428528860954426, 0.6068316106809032
# theta_0 for Pegasos is -1.2321908391886922

# T = 120000
# theta for Pegasos is 0.6427442693361588, 0.6068624564464895
# theta_0 for Pegasos is -1.2321883787748875

# T = 110000




# Fix L at 0.001, Tune T
# RESULTS
# perceptron valid: [(1, 0.56), (5, 0.778), (10, 0.752), (15, 0.758), (25, 0.766), (50, 0.766)]
# best = 0.7780, T=5.0000
# avg perceptron valid: [(1, 0.704), (5, 0.762), (10, 0.782), (15, 0.776), (25, 0.786), (50, 0.786)]
# best = 0.7860, T=25.0000
# Pegasos valid: tune T [(1, 0.648), (5, 0.67), (10, 0.764), (15, 0.776), (25, 0.788), (50, 0.788)]
# best = 0.7880, T=25.0000
# Pegasos valid: tune L [(0.001, 0.788), (0.01, 0.766), (0.1, 0.602), (1, 0.558), (10, 0.504)]
# best = 0.7880, L=0.0010


# Fix L at 0.01, Tune T
# RESULTS
# perceptron valid: [(1, 0.56), (5, 0.778), (10, 0.752), (15, 0.758), (25, 0.766), (50, 0.766)]
# best = 0.7780, T=5.0000
# avg perceptron valid: [(1, 0.704), (5, 0.762), (10, 0.782), (15, 0.776), (25, 0.786), (50, 0.786)]
# best = 0.7860, T=25.0000
# Pegasos valid: tune T [(1, 0.586), (5, 0.758), (10, 0.756), (15, 0.764), (25, 0.766), (50, 0.77)]
# best = 0.7700, T=50.0000
# Pegasos valid: tune L [(0.001, 0.788), (0.01, 0.77), (0.1, 0.742), (1, 0.57), (10, 0.502)]
# best = 0.7880, L=0.0010

# Fix L at 0.1, Tune T
# RESULTS

# perceptron valid: [(1, 0.56), (5, 0.778), (10, 0.752), (15, 0.758), (25, 0.766), (50, 0.766)]
# best = 0.7780, T=5.0000
# avg perceptron valid: [(1, 0.704), (5, 0.762), (10, 0.782), (15, 0.776), (25, 0.786), (50, 0.786)]
# best = 0.7860, T=25.0000
# Pegasos valid: tune T [(1, 0.532), (5, 0.56), (10, 0.584), (15, 0.592), (25, 0.602), (50, 0.742)]
# best = 0.7420, T=50.0000
# Pegasos valid: tune L [(0.001, 0.788), (0.01, 0.77), (0.1, 0.742), (1, 0.57), (10, 0.502)]
# best = 0.7880, L=0.0010


# Fix L at 1, Tune T
# RESULTSx``

# perceptron valid: [(1, 0.56), (5, 0.778), (10, 0.752), (15, 0.758), (25, 0.766), (50, 0.766)]
# best = 0.7780, T=5.0000
# avg perceptron valid: [(1, 0.704), (5, 0.762), (10, 0.782), (15, 0.776), (25, 0.786), (50, 0.786)]
# best = 0.7860, T=25.0000
# Pegasos valid: tune T [(1, 0.596), (5, 0.508), (10, 0.526), (15, 0.508), (25, 0.558), (50, 0.57)]
# best = 0.5960, T=1.0000
# Pegasos valid: tune L [(0.001, 0.648), (0.01, 0.586), (0.1, 0.532), (1, 0.596), (10, 0.476)]
# best = 0.6480, L=0.0010



# Fix L at 10, Tune T
# RESULTS
# perceptron valid: [(1, 0.56), (5, 0.778), (10, 0.752), (15, 0.758), (25, 0.766), (50, 0.766)]
# best = 0.7780, T=5.0000
# avg perceptron valid: [(1, 0.704), (5, 0.762), (10, 0.782), (15, 0.776), (25, 0.786), (50, 0.786)]
# best = 0.7860, T=25.0000
# Pegasos valid: tune T [(1, 0.476), (5, 0.488), (10, 0.494), (15, 0.498), (25, 0.504), (50, 0.502)]
# best = 0.5040, T=25.0000
# Pegasos valid: tune L [(0.001, 0.788), (0.01, 0.766), (0.1, 0.602), (1, 0.558), (10, 0.504)]
# best = 0.7880, L=0.0010


# # values of T and lambda to try
# Ts = [1, 5, 10, 15, 25, 50]
# Ls = [0.001, 0.01, 0.1, 1, 10]

# pct_tune_results = utils.tune_perceptron(Ts, *data)
# print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

# avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
# print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# # fix values for L and T while tuning Pegasos T and L, respective
# # fix_L = 10
# fix_L = 10
# peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
# print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
# print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

# # fix_T = Ts[np.argmax(peg_tune_results_T[1])]
# fix_T = Ts[np.argmax(peg_tune_results_T[1])]
# peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
# print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
# print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
# utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)



# ===T fixed at T=1===
# perceptron valid: [(1, 0.56), (5, 0.778), (10, 0.752), (15, 0.758), (25, 0.766), (50, 0.766)]
# best = 0.7780, T=5.0000
# avg perceptron valid: [(1, 0.704), (5, 0.762), (10, 0.782), (15, 0.776), (25, 0.786), (50, 0.786)]
# best = 0.7860, T=25.0000
# Pegasos valid: tune L [(0.001, 0.648), (0.01, 0.586), (0.1, 0.532), (1, 0.596), (10, 0.476)]
# best = 0.6480, L=0.0010

# Pegasos valid: tune T [(1, 0.648), (5, 0.67), (10, 0.764), (15, 0.776), (25, 0.788), (50, 0.788)]
# best = 0.7880, T=25.0000



# !!!!!!!!!!!!!!!!!!!!!!T fixed at === 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# perceptron valid: [(1, 0.758), (5, 0.72), (10, 0.716), (15, 0.778), (25, 0.794), (50, 0.79)]
# best = 0.7940, T=25.0000
# avg perceptron valid: [(1, 0.794), (5, 0.792), (10, 0.798), (15, 0.798), (25, 0.8), (50, 0.796)]
# best = 0.8000, T=25.0000
# Pegasos valid: tune L [(0.001, 0.776), (0.01, 0.786), (0.1, 0.732), (1, 0.598), (10, 0.518)]
# best = 0.7860, L=0.0100

# Pegasos valid: tune T [(1, 0.786), (5, 0.78), (10, 0.79), (15, 0.802), (25, 0.806), (50, 0.8)]
# best = 0.8060, T=25.0000



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ===T fixed at === 5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# perceptron valid: [(1, 0.758), (5, 0.72), (10, 0.716), (15, 0.778), (25, 0.794), (50, 0.79)]
# best = 0.7940, T=25.0000
# avg perceptron valid: [(1, 0.794), (5, 0.792), (10, 0.798), (15, 0.798), (25, 0.8), (50, 0.796)]
# best = 0.8000, T=25.0000
# Pegasos valid: tune L [(0.001, 0.8), (0.01, 0.78), (0.1, 0.75), (1, 0.616), (10, 0.518)]
# best = 0.8000, L=0.0010

# Pegasos valid: tune T [(1, 0.776), (5, 0.8), (10, 0.802), (15, 0.794), (25, 0.786), (50, 0.79)]
# best = 0.8020, T=10.0000


# ===!!!!!!!!!!!!!!!!!!!!!!!!T fixed at === 10 !!!!!!!!!!!!!!!!
# perceptron valid: [(1, 0.758), (5, 0.72), (10, 0.716), (15, 0.778), (25, 0.794), (50, 0.79)]
# best = 0.7940, T=25.0000
# avg perceptron valid: [(1, 0.794), (5, 0.792), (10, 0.798), (15, 0.798), (25, 0.8), (50, 0.796)]
# best = 0.8000, T=25.0000

# Pegasos valid: tune L [(0.001, 0.802), (0.01, 0.79), (0.1, 0.752), (1, 0.586), (10, 0.518)]
# best = 0.8020, L=0.0010
# Pegasos valid: tune T [(1, 0.776), (5, 0.8), (10, 0.802), (15, 0.794), (25, 0.786), (50, 0.79)]
# best = 0.8020, T=10.0000


# ===!!!!!!!!!!!!!!!!!!!!!!!!T fixed at === !!!!!!!!!!!!!!!!15
# perceptron valid: [(1, 0.758), (5, 0.72), (10, 0.716), (15, 0.778), (25, 0.794), (50, 0.79)]
# best = 0.7940, T=25.0000
# avg perceptron valid: [(1, 0.794), (5, 0.792), (10, 0.798), (15, 0.798), (25, 0.8), (50, 0.796)]
# best = 0.8000, T=25.0000

# Pegasos valid: tune L [(0.001, 0.794), (0.01, 0.802), (0.1, 0.76), (1, 0.594), (10, 0.518)]
# best = 0.8020, L=0.0100
# Pegasos valid: tune T [(1, 0.786), (5, 0.78), (10, 0.79), (15, 0.802), (25, 0.806), (50, 0.8)]
# best = 0.8060, T=25.0000


# ===!!!!!!!!!!!!!!!!!!!!!!!!T fixed at === 15 !!!!!!!!!!!!!!!!
# perceptron valid: [(1, 0.758), (5, 0.72), (10, 0.716), (15, 0.778), (25, 0.794), (50, 0.79)]
# best = 0.7940, T=25.0000
# avg perceptron valid: [(1, 0.794), (5, 0.792), (10, 0.798), (15, 0.798), (25, 0.8), (50, 0.796)]
# best = 0.8000, T=25.0000

# Pegasos valid: tune L [(0.001, 0.794), (0.01, 0.802), (0.1, 0.76), (1, 0.594), (10, 0.518)]
# best = 0.8020, L=0.0100
# Pegasos valid: tune T [(1, 0.786), (5, 0.78), (10, 0.79), (15, 0.802), (25, 0.806), (50, 0.8)]
# best = 0.8060, T=25.0000




# reviews = 
# train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
# print(extract_bow_feature_vectors(reviews, indices_by_word, binarize=True))
# theta = np.zeros(2,)
# theta_0 = 0

# print(classify(train_feature_matrix, theta, theta_0))
# print(accuracy(preds, target), 'hello')






# def classify(feature_matrix, theta, theta_0):
#     """
#     A classification function that uses given parameters to classify a set of
#     data points.

#     Args:
#         `feature_matrix` - numpy matrix describing the given data. Each row
#             represents a single data point.
#         `theta` - numpy array describing the linear classifier.
#         `theta_0` - real valued number representing the offset parameter.

#     Returns:
#         a numpy array of 1s and -1s where the kth element of the array is the
#         predicted classification of the kth row of the feature matrix using the
#         given theta and theta_0. If a prediction is GREATER THAN zero, it
#         should be considered a positive classification.
#     """
#     # Your code here
#     # print(feature_matrix.shape[0])
#     lists =  np.zeros(feature_matrix.shape[0])
#     epsilon =  np.finfo(np.float32).eps
#     for i in range(feature_matrix.shape[0]):
#         if np.dot(theta, feature_matrix[i]) + theta_0 <= epsilon:
#             lists.append(-1)
#         else:
#             lists.append(1)
#     return np.array(lists)
            
#     # raise NotImplementedError

# print(classify(train_feature_matrix,theta,theta_0))

    #     # print(feature_matrix)
    #     # return feature_matrix
    #     for i, text in enumerate(reviews):
    #         word_list = extract_words(text)
    #     for word in word_list:
    #         if word in indices_by_word:
    #             feature_matrix[i, indices_by_word[word]] = 1
    #     print(feature_matrix)
    # print(feature_matrix)






    