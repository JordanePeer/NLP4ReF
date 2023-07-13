##################################################################################################################
"""
Import the necessary libraries and modules for the code.
"""

# Step 1:  Import necessary libraries and modules, Pickle, Numpy, and Pandas  
import pickle
import numpy as np
import pandas as pd

# Step 2: Import specific modules for classification from the Scikit-Learn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Step 3: Import specific modules for natural language processing from the NLTK module
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Step 4: Import specific modules for word embeddings
from collections import defaultdict
from gensim.models import Word2Vec
stopwords = set(stopwords.words('english'))

##################################################################################################################
def upload_module(file):
    """
    Uploads the data according to a certain template.
    """
    
    # Step 1: Read the CSV file and Skip the header row
    df = pd.read_csv(file, sep=',', header=0, quotechar = '"', doublequote=True)

    # Step 2: Initialize empty lists for each column
    requirements_numbers, requirements, frnfr_categories, system_classes = [], [], [], []

    # Step 3: Iterate through each row of the CSV file
    for i, row in df.iterrows():
        
        # Step 4: Append the value of each column to the corresponding list
        requirements_numbers.append(row['Requirement Number'])
        requirements.append(row['Requirement'])
        frnfr_categories.append(row['FR/NFR Category'])
        system_classes.append(row['System Class'])

    # Step 5: return the lists to the main script
    return(requirements_numbers, requirements, frnfr_categories, system_classes)


##################################################################################################################
def normalization_module(requirements):
    """
    Normalize a list of requirements by removing stop words, lemmatizing words, and joining them into a string.
    """

    # Step 1: Initialize the WordNet lemmatizer and Define a tag map for lemmatization
    lemmatizer = WordNetLemmatizer()
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    # Step 2: Initialize a list to store the filtered requirements
    filtered_requirements = []

    # Step 3: Normalize each requirement in the list
    for requirement in requirements:
        # Step 4: Tokenize the requirement: Convert to lowercase and tokenize 
        tokens = word_tokenize(requirement.lower())
        
        # Step 5: Remove the stop words and lemmatize the remaining words
        filtered_tokens = [lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(tokens) 
                           if token.isalpha() and token not in stopwords]

        # Step 6: Join the filtered tokens back into a string
        filtered_requirement = ' '.join(filtered_tokens)

        # Step 7: Append the filtered requirement to the list
        filtered_requirements.append(filtered_requirement)
    
    # Step 8: Return the normalized requirements
    return filtered_requirements


##################################################################################################################
def get_splits(filtered_requirements, y_list):
    """
    Split the data into Test and Training lists with a ratio of 0.2, just for requirements generation tests.
    """

    # Step 1: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(filtered_requirements, y_list,
                                                        test_size=0.2, random_state=42)
    
    # Step 2: Return the split requirements sets
    return X_train, X_test



##################################################################################################################
def vectorization_module(filtered_requirements):
    """
    Vectorize the list of normalized requirements using the TF-IDF technique.
    """

    # Step 1: Initialize the TF-IDF vectorizer (the object that turns text into vectors)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')

    # Step 2: Fit and transform the filtered_requirements to create the feature matrix
    dtm = vectorizer.fit_transform(filtered_requirements)

    # Step 3: Store the trained classifier for production use
    vec_filename = 'FRNFR_vectoriser.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))    
    
    # Step 4: Return the feature matrix and feature names as a tuple
    return (dtm, vectorizer)


##################################################################################################################
def Encoder_Y_module(class_names, vec_filename):
    """
    Encoder the list of FR/NFR requirements labels using the Sklearn Label Encoder technique.
    """

    # Step 1: Initialize the Y Label Encoder (the object that turns text into vectors)
    label_encoder = LabelEncoder()
    
    # Step 2: Fit and transform the requirements labels
    label_encoder.fit(class_names)

    # Step 3: Store the trained classifier for production using
    pickle.dump(label_encoder, open(vec_filename, 'wb'))    
    
    # Step 4: Return the encoder
    return (label_encoder)



##################################################################################################################
def evaluate_classifier(classifier, vectoriser, X_t, y_t):
    """
    Evaluates the performance of a classifier on the test set.
    """
    
    # Step 1: Transform the data using the vectorizer
    X_t_tfidf = vectoriser.transform(X_t)
    
    # Step 2: Predict labels for the transformed data
    y_pred = classifier.predict(X_t_tfidf)

    # Step 3: Calculate precision, recall, and F1-score metrics
    metrics_prf = precision_recall_fscore_support(y_t, y_pred, average='weighted')

    # Step 4: Return the metrics as a list
    return [metrics_prf[0], metrics_prf[1], metrics_prf[2]]



##################################################################################################################  
def train_FRNFR_classifier(vectorizer, label_encoder, requirements_numbers, filtered_requirements, Y_categories):
    """
    Classify the text data using a multi-task SVM model with Chi-squared Feature Selection, 
    for the FR/NFR classification.
    """

    # Step 1: Initialize variables to store training and test results
    training_results = [0, 0, 0]
    test_results = [0, 0, 0]
    
    # Step 2: Number of iterations for cross-validation (K = 1 multiple numbers of categories)
    K = 12

    # Step 3: Perform stratified group 12-fold cross-validation
    skf = StratifiedKFold(n_splits=K, random_state=70, shuffle=True)
    X = np.array(filtered_requirements)
    y = np.array(Y_categories)
    groups = np.array(requirements_numbers)
    
    # Step 4: Perform K-fold cross-validation
    for train_index, test_index in skf.split(X, y, groups):
        # Step 5: Organize the datasets per their split indexes
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Step 6: Encode and Vectorize the datasets       
        dtm_train = vectorizer.transform(X_train)
        label_train = label_encoder.transform(y_train)
        label_test = label_encoder.transform(y_test)   
         
        # Step 7: Train a Support Vector Machine (SVM) classifier 
        # IoT - 2.5 (80) - Test: precision = 0.7283648043237546; recall = 0.71875; f1_score = 0.7177258243346439
        # exp - 1.5 (70) - Test: precision = 0.7658004963784767; recall = 0.7605195473251029; f1_score = 0.7533329283171867
        SVM_classifier = SVC(C=1.5, kernel='linear', class_weight="balanced").fit(dtm_train, label_train) 

        # Step 8: Evaluate the classifier on training data
        Train_result = evaluate_classifier(SVM_classifier, vectorizer, X_train, label_train)
        for i in range(len(training_results)):
            training_results[i] += Train_result[i]
            
        # Step 9: Evaluate the classifier on testing data
        Test_result = evaluate_classifier(SVM_classifier, vectorizer, X_test, label_test)
        for i in range(len(test_results)):
            test_results[i] += Test_result[i]
            
    # Step 10: Calculate average results over K iterations
    for i in range(len(test_results)):
        training_results[i] /= K
        test_results[i] /= K
    
    # Step 11: Store the trained classifier for production use
    clf_filename = 'SVM_FRNFR_classifier.pkl'
    pickle.dump(SVM_classifier, open(clf_filename, 'wb'))

    # Step 12: Print the average training and testing results
    line = "Training: precision = {}; recall = {}; f1_score = {}".format(training_results[0], training_results[1], 
                                                                         training_results[2])
    print(line)

    line = 'Test: precision = {}; recall = {}; f1_score = {}'.format(test_results[0], test_results[1], 
                                                                     test_results[2])
    print(line)


##################################################################################################################
def train_Sys_classifier(vectorizer, label_encoder, requirements_numbers, filtered_requirements, Y_categories):
    """
    Classify the text data using a multi-task SVM model with Chi-squared Feature Selection, 
    for the classification of the system.
    """

    # Step 1: Initialize variables to store training and test results
    training_results = [0, 0, 0]
    test_results = [0, 0, 0]
    
    # Step 2: Number of iterations for cross-validation (K = 2 multiple numbers of categories)
    K = 12

    # Step 3: Perform stratified group 12-fold cross-validation
    skf = StratifiedKFold(n_splits=K, random_state=24, shuffle=True)
    X = np.array(filtered_requirements)
    y = np.array(Y_categories)
    groups = np.array(requirements_numbers)
    
    # Step 4: Perform K-fold cross-validation
    for train_index, test_index in skf.split(X, y, groups):
        # Step 5: Organize the datasets per their split indexes
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Step 6: Encode and Vectorize the datasets       
        dtm_train = vectorizer.transform(X_train)
        label_train = label_encoder.transform(y_train)
        label_test = label_encoder.transform(y_test)   
         
        # Step 7: Train a Support Vector Machine (SVM) classifier
        # IoT - 1.7 - Test: precision = 0.8365097379063339; recall = 0.8293125; f1_score = 0.8283905718639684
        SVM_classifier = SVC(C=1.7, kernel='linear', class_weight="balanced").fit(dtm_train, label_train) 
                
        # Step 8: Evaluate the classifier on training data
        Train_result = evaluate_classifier(SVM_classifier, vectorizer, X_train, label_train)
        for i in range(len(training_results)):
            training_results[i] += Train_result[i]
            
        # Step 9: Evaluate the classifier on testing data
        Test_result = evaluate_classifier(SVM_classifier, vectorizer, X_test, label_test)
        for i in range(len(test_results)):
            test_results[i] += Test_result[i]
            
    # Step 10: Calculate average results over K iterations
    for i in range(len(test_results)):
        training_results[i] /= K
        test_results[i] /= K

    # Step 11: Store the trained classifier for production use
    clf_filename = 'SVM_Sys_classifier.pkl'
    pickle.dump(SVM_classifier, open(clf_filename, 'wb'))

    # Step 12: Print the average training results
    line = "Training: precision = {}; recall = {}; f1_score = {}".format(training_results[0], training_results[1], 
                                                                         training_results[2])
    print(line)
    
    # Step 13: Print the average test results
    line = 'Test: precision = {}; recall = {}; f1_score = {}'.format(test_results[0], test_results[1], 
                                                                     test_results[2])
    print(line)
    

##################################################################################################################
def sys_list_arrangement(system_classes, requirements):
    """ 
    Organizes the requirements based on their corresponding system classes, and 
    returns a dictionary containing lists of requirements for each system class.
    """

    # Step 1: Create a dictionary to hold the system class lists
    system_class_lists = {}

    # Step 2: Iterate over the system_classes list
    for i, system_class in enumerate(system_classes):
        
        # Step 3: Check if the system class list exists in the dictionary, create it if not
        if system_class not in system_class_lists:
            system_class_lists[system_class] = []
    
        # Step 4: Add the requirement to the system class list
        system_class_lists[system_class].append(requirements[i])
    
    # Step 5: Return the system_class_lists to the main script
    return system_class_lists



##################################################################################################################
def sentence_to_vector(req, model):
    """ 
    Converts a sentence into a vector representation using a Word2Vec model.
    - For each token in the sentence, retrieve the word vector from the Word2Vec model.
    - If the token is not present in the model's vocabulary, it is skipped.
    - Finally, take the mean of all the word vectors to get the sentence vector.
    """
    
    # Step 1: Tokenization
    tokens = req.lower().split()
    
    # Step 2: Vector Calculation
    vector = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
    
    # Step 3: Return the requirement vector
    return vector



##################################################################################################################
def generate_requirements(requirements, system_classes):
    """ 
    Generate a list of closest requirements based on similarity using Word2Vec embeddings.
    """
    
    # Step 1: Splitting the dataset into training and testing sets
    X_train, X_test = get_splits(requirements, system_classes)

    # Step 2: Training the Word2Vec model
    tokenized_train_dataset = [req.lower().split() for req in X_train]
    model = Word2Vec(tokenized_train_dataset, min_count=1)
    
    # Step 3: Saving the trained model to disk
    model.save("word2vec_model.bin")
    
    # Step 4: Saving the database to disk   
    clf_filename = 'Generative_dataset.pkl'
    pickle.dump(X_train, open(clf_filename, 'wb'))

    # Step 5: Computing vectors for the training set
    train_vectors = [sentence_to_vector(req, model) for req in X_train]
             
    # Step 6: Setting the batch size for similarity calculations
    batch_size = 100
    
    # Step 7: Initializing the list to store the new requirements
    new_requirements = []
    
    # Step 8: Computing similarities in batches
    for i in range(0, len(X_test), batch_size):
        
        # Step 9: Extracting a batch of requirements for similarity calculation
        batch_reqs = X_test[i:i+batch_size]
        batch_vectors = [sentence_to_vector(req, model) for req in batch_reqs]
        
        # Step 10: Calculating cosine similarities between batch vectors and training vectors
        similarities = np.dot(batch_vectors, np.transpose(train_vectors))
        similarities /= np.outer(np.linalg.norm(batch_vectors, axis=1), np.linalg.norm(train_vectors, axis=1))
        
        # Step 11: Finding the indices of the two closest requirements for each batch requirement
        closest_indices = np.argsort(similarities, axis=1)[:, -2:]
        closest_reqs = [req for indices in closest_indices for req in [X_train[idx] for idx in indices]]
        
        # Step 12: Adding the closest requirements to the new_requirements list
        new_requirements.extend(closest_reqs)
        
    # Step 13: Removing duplicate or identical requirements from the new_requirements list
    new_requirements = list(set(new_requirements))
    
    # Step 14: Returning the list of closest requirements
    return new_requirements



##################################################################################################################
def new_file_module(requirements_numbers, requirements):
    """
    Creates a new CSV file with the new requirement data.
    """
    
    # Step 1: Create an empty DataFrame and define the columns
    df = pd.DataFrame(columns=["Requirement Number", "Requirement", "System Class", "FR/NFR Category"])

    # Step 2: Iterate over the requirement data
    for i in range(len(requirements)):
        row = {"Requirement Number": requirements_numbers[i], "Requirement": requirements[i]}
        
        # Step 3: Append the new row to the DataFrame
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Step 4: Write the DataFrame to a CSV file
    df.to_csv("New_requirements.csv", index=False)

    # Step 5: Return the CSV file path
    print("Your new file is test.csv")




##################################################################################################################
# Step 1: Print the initial Greetings message
print("Greetings Systems Engineer, The NLP4ReF-NLTK will categorize and generate requirements, based on the initial files you provided")

# Step 2: Upload initial files 
file = 'NLP4ReF-NLTK/Initial_Files/PROMISE_exp.csv'
requirements_numbers, requirements, frnfr_categories, system_classes = upload_module(file)

# Step 3: Perform natural language normalization on requirements
filtered_requirements = normalization_module(requirements)

# Step 4: Split the requirements and categories into training and testing sets
X_train, X_test = get_splits(filtered_requirements, frnfr_categories)

# Step 5: Vectorize the training data
dtm, vectorizer = vectorization_module(filtered_requirements)
label_FRNFR_encoder = Encoder_Y_module(['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO'], 'FRNFR_Labels_Encoder.pkl')
label_Sys_encoder = Encoder_Y_module(['SE', 'AC', 'CO', 'DS', 'SW', 'UI'], 'System_Labels_Encoder.pkl')

# Step 6: Train the FR/NFR classifier
print("FR / NFR")
train_FRNFR_classifier(vectorizer, label_FRNFR_encoder, requirements_numbers, filtered_requirements, frnfr_categories)

# Step 7: Train the system class classifier
print("Systems")
train_Sys_classifier(vectorizer, label_Sys_encoder, requirements_numbers, filtered_requirements, system_classes)

# Step 8: Generate new requirements based on initial files
delta_list = generate_requirements(requirements, system_classes)

# Step 9: Creates a new CSV file with the new requirement data
new_file_module(requirements_numbers, delta_list)

# Step 10: Print completion message
print("the Developement phase is done")
