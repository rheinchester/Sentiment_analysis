import numpy as np
def hinge_loss():
    pass

def hinge_loss_single(feature_vector, label, theta):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    z = label - (np.dot(feature_vector, theta))
    print( label - (np.dot(feature_vector, theta)) , 'hinge_loss)_single')
    if(z >= 1):
        return 0
    else:
        return 1 - z
    
def squared_loss(feature_vector, label, theta):
    # z = label - (np.dot(feature_vector, theta))
    z = label - (np.dot(feature_vector, theta))
    print(z, (z**2)/2)
    return (z**2)/2

def emprical_risk(feature_matrix, labels, theta, hinge=False, sq_error=False):
    if hinge:
        hinge_loss_total = 0
        for i in range(len(labels)):
            hinge_loss_total+= hinge_loss_single(feature_matrix[i], labels[i], theta)
            # print(hinge_loss_single(feature_matrix[i], labels[i], theta))
        return hinge_loss_total/len(labels)
    if sq_error:
        sq_err_total = 0
        for i in range(len(labels)):
            sq_err_total+=squared_loss(feature_matrix[i], labels[i], theta)
        return sq_err_total/len(labels)
    
feature_matrix = [
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, -1],
    [-1, 1, 1],
]
labels = [2, 2.7, -0.7, 2]
theta = [0, 1, 2]
print(emprical_risk(feature_matrix, labels, theta, sq_error=True))
