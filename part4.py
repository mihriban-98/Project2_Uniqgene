def reverse_onehot(y_vals, num_lab):

    y_vals_np = y_vals.numpy()
    y_vals_rev = np.empty([len(y_vals_np),1]) #y_test length

    for i in range(len(y_vals_np)):
        for j in range(num_lab): 
            if y_vals_np[i, j] == 1:
                y_vals_rev[i,0] = j
    return y_vals_rev

y_test_rev = reverse_onehot(y_test,7)
y_predicted2_rev = reverse_onehot(y_predicted2,7)
#print(y_test_rev)

#Results for different epochs

# epoch 400 lr = 0.1

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

#confusion_matrix(y_test_rev,y_predicted2_rev)

# epoch 300 lr = 0.1

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

#confusion_matrix(y_test_rev,y_predicted2_rev)
#print(confusion_matrix(y_test_rev,y_predicted2_rev))

# epoch 400 lr = 0.01

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

#confusion_matrix(y_test_rev,y_predicted2_rev)
#print(confusion_matrix(y_test_rev,y_predicted2_rev))
