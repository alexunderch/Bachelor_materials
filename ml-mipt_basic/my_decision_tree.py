import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
   
    H = 0.
    for c in range(y.shape[1]):
        p_k = np.sum(y.T[c]) / len(y) 
        H -= float(p_k * np.log(p_k + EPS))
    return H
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE 
    H = 1.
    for c in range(y.shape[1]):
        p_k = float(np.sum(y.T[c]) / len(y))
        H -= p_k ** 2
    return H 

def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    return np.mean((y - np.mean(y)) ** 2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    return np.mean(np.abs(y - np.median(y))) 
    

def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype = float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis = 1)[:, None]


class Leaf:
    """
    A Leaf node classifies data.

    This holds a dictionary of number of times the
    class appears in the rows from the training data that reach this leaf.
    """
    def __init__(self, y, classification = True, criterion_name = None):
        def count(y):
            """Counts the number of each type of example in a dataset"""
            counts = {}  
            y = one_hot_decode(y)
            for label in y.T[0]:
                if label not in counts: counts[label] = 0
                counts[label] += 1
            return counts
            
        if classification: self.predictions = count(y)
        else: 
            if criterion_name == 'mad_median':
                self.predictions = np.median(y)
            if criterion_name == 'variance': self.predictions = np.mean(y)


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba = 0.):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None    
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes = None, max_depth = np.inf, min_samples_split = 2, 
                 criterion_name = 'gini', debug = False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

            
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE

        X_left, y_left = X_subset[X_subset[:, feature_index] < threshold], y_subset[np.where(X_subset[:, feature_index] < threshold)] 
        X_right, y_right = X_subset[X_subset[:, feature_index] >= threshold], y_subset[np.where(X_subset[:, feature_index] >= threshold)] 
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """ 
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        y_left = y_subset[np.where(X_subset[:, feature_index] < threshold)] 
        y_right = y_subset[np.where(X_subset[:, feature_index] >= threshold)] 
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE

        def compute_gain(y_left, y_right, y_subset):
            """
            Computes weighted criterion of split

            Parameters:
            ----------
            y_true: np.array of type float with shape (n_objects, 1)
            One-hot representation of class labels or target values for corresponding subset

            Returns:
            -------
            float
            Criterion of the provided split
            """
            self.criterion = self.all_criterions[self.criterion_name][0]
            p = len(y_left) / len(y_subset)
            return self.criterion(y_subset) - (p * self.criterion(y_left) + (1 - p) * self.criterion(y_right)) 

        max_criterion = 0.
        feature_index, threshold = 0, 0.
        _, n_features = X_subset.shape

        for feature in range(n_features):
            featured_column = set(X_subset.T[feature])
            for thr in featured_column:

                y_left, y_right = self.make_split_only_y(feature, thr, X_subset, y_subset)
                if len(y_left) == 0 or len(y_right) == 0: continue

                curr_criterion = compute_gain(y_left, y_right, y_subset)
                if curr_criterion > max_criterion:
                    max_criterion = curr_criterion
                    threshold = thr; feature_index = feature

        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        def compute_split(node, X_subset, y_subset, depth):
            """
            Recursevely splits given node:
            Parameters:
            ---------------
            node: root node to split from (Node class)
             X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

            y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset 

            Returns:
            ---------------
            a fitted tree with root node
            """
            
            (X_left, y_left), (X_right, y_right) = self.make_split(node.feature_index, 
                                                        node.value, X_subset, y_subset)
            n_objects = len(y_subset.copy())
            del(X_subset); del(y_subset)

            if len(X_left) == 0 and len(X_right) == 0:    
                node.left_child = node.right_child = Leaf(np.concatenate([y_left, y_right]), \
                self.classification, self.criterion_name)
                return 

            if depth >= self.max_depth: 
                node.left_child = Leaf(y_left, self.classification, self.criterion_name)
                node.right_child = Leaf(y_right, self.classification, self.criterion_name)
                return 

            if len(X_left) <= self.min_samples_split:
                node.left_child = Leaf(y_left, self.classification, self.criterion_name) 
            else:
                lfi, lthr = self.choose_best_split(X_left, y_left)
                node.left_child = Node(lfi, lthr, proba = len(X_left) / n_objects)

                compute_split(node.left_child, X_left, y_left, depth + 1)

            if len(X_right) <= self.min_samples_split: 
                node.right_child = Leaf(y_right, self.classification, self.criterion_name) 
            else:
                rfi, rthr = self.choose_best_split(X_right, y_right)
                node.right_child = Node(rfi, rthr, proba = len(X_right) / n_objects)

                compute_split(node.right_child, X_right, y_right, depth + 1)

        #creating root
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        root_node = Node(feature_index, threshold)
        
        compute_split(root_node, X_subset.copy(), y_subset.copy(), depth = 1)

        self.depth = self.max_depth

        return root_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        # YOUR CODE HERE
        def compute_prediction(row, node = self.root):
            """ 
            Recursevly computes predictions, comparing obj values with threshold
            """
            if row[node.feature_index] < node.value:
                if isinstance(node.left_child, Leaf):
                    if self.classification: return max(node.left_child.predictions, \
                    key = lambda x: node.left_child.predictions[x])
                    return node.left_child.predictions
                elif isinstance(node.left_child, Node):
                        return compute_prediction(row, node.left_child)
            else:
                if isinstance(node.right_child, Leaf):
                    if self.classification: return max(node.right_child.predictions, \
                    key = lambda x: node.right_child.predictions[x])
                    return node.right_child.predictions
                elif isinstance(node.right_child, Node): 
                    return compute_prediction(row, node.right_child)

        y_predicted = list()
        for x in X: y_predicted.append(compute_prediction(x))

        return y_predicted 

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        def compute_proba(row, node = self.root):
            if row[node.feature_index] < node.value:
                if isinstance(node.left_child, Leaf):
                    return  {k: v / total for total in \
                    (sum(node.left_child.predictions.values(), 0.0),) for k, v in node.left_child.predictions.items()}
                elif isinstance(node.left_child, Node):
                    return compute_proba(row, node.left_child)
            else:
                if isinstance(node.right_child, Leaf):
                    return  {k: v / total for total in \
                    (sum(node.right_child.predictions.values(), 0.0),) for k, v in node.right_child.predictions.items()}
                elif isinstance(node.right_child, Node): 
                    return compute_proba(row, node.right_child)


        y_predicted_probs = np.zeros((len(X), self.n_classes))
        for ind in range(len(X)):
            obj = compute_proba(X[ind])
            # print(obj)
            probs = np.zeros((1, self.n_classes))
            for c in range(self.n_classes):
                if c not in obj: probs[:, c] = 0.
                else: probs[:, c] = obj[c]

            y_predicted_probs[ind] = probs
        assert (len(y_predicted_probs) == len(X))

        # YOUR CODE HERE
        
        return y_predicted_probs
