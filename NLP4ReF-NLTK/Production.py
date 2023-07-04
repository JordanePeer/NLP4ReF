##################################################################################################################
# Step 1:  Importing necessary libraries and modules, Os, Pickle, Numpy and Pandas  
import os
import pickle
import numpy as np
import pandas as pd

# Step 2: Importing specific modules for natural language processing from the NLTK module
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, stopwords

# Step 3: Importing specific modules for word embeddings
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
    return requirements_numbers, requirements, frnfr_categories, system_classes



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
def Classification_module(requirements, frnfr_categories, system_classes):
    """
    Classify requirements into FR/NFR categories and system classes.
    """

    # Step 1: Load the FR/NFR vectorizer
    vec_filename = 'FRNFR_vectoriser.pkl'
    FRNFR_vec = pickle.load(open(vec_filename, 'rb'))

    # Step 2: Load the FR/NFR classifier
    clf_filename = 'SVM_FRNFR_classifier.pkl'
    FRNFR_clf = pickle.load(open(clf_filename, 'rb'))

    # Step 3: Load the system class vectorizer
    vec_filename = 'SYS_vectoriser.pkl'
    SYS_vec = pickle.load(open(vec_filename, 'rb'))

    # Step 4: Load the the system class classifier
    clf_filename = 'SVM_Sys_classifier.pkl'
    SYS_clf = pickle.load(open(clf_filename, 'rb'))

    # Step 5: Vectorize and classify the requirements, into FR/NFR categories and system classes
    pred_FRNFR = FRNFR_clf.predict(FRNFR_vec.transform(requirements))
    pred_SYS = SYS_clf.predict(SYS_vec.transform(requirements))

    # Step 6: Insert the results inside their lists
    frnfr_categories, system_classes = list(pred_FRNFR), list(pred_SYS)

    # Step 7: retrurn the lists of FR/NFR categories and system classes to the main script
    return frnfr_categories, system_classes




##################################################################################################################
def new_file_module(requirements_numbers, requirements, system_classes, frnfr_categories):
    """
    Creates a new CSV file with the new requirement data.
    """
    
    # Step 1: Create an empty DataFrame and define the columns
    df = pd.DataFrame(columns=["Requirement Number", "Requirement", "System Class", "FR/NFR Category"])

    # Step 2: Iterate over the requirement data
    for i in range(len(requirements_numbers)):
        row = {"Requirement Number": requirements_numbers[i], "Requirement": requirements[i], 
               "System Class": system_classes[i], "FR/NFR Category": frnfr_categories[i]}
        
        # Step 3: Append the new row to the DataFrame
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Step 4: Save the DataFrame as a CSV file
    df.to_csv("Initial_Classify_Requirements.csv", index=False)

    # Step 5: Print a confirmation message with the CSV file path
    print("Your new file is Initial_Classify_Requirements.csv")



##################################################################################################################
def sys_list_arrangement(system_classes, requirements):
    """ 
    Arrange into separate lists based on their associated system classes.   
    """

    # Step 1: Initialize an empty dictionary to store system class lists
    system_class_lists = {}

    # Step 2: Iterate over the system classes and requirements
    for i, system_class in enumerate(system_classes):

        # Step 3: Check if the system class is already present in the dictionary
        if system_class not in system_class_lists:
            system_class_lists[system_class] = []
    
        # Step 4: Append the corresponding requirement to the system class list
        system_class_lists[system_class].append(requirements[i])
    
    # Step 5: Return the dictionary with system class lists
    return system_class_lists




###################################################################################################
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
def Forecaste_module(system_class_lists):
    """
    Generate a list of closest requirements based on similarity using Word2Vec embeddings.
    """
    
    # Step 1: Loading the saved model and training dataset from disk
    model = Word2Vec.load("word2vec_model.bin")
    with open("Generative_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    # Step 2: Computing vectors for the training set
    train_vectors = [sentence_to_vector(req, model) for req in dataset]
    
    # Step 3: Initializing the list to store the new requirements
    new_requirements = []
    
    # Step 4: Initializing the delta_list
    delta_list = {}
            
    # Step 6: Setting the batch size for similarity calculations
    batch_size = 100  

    # Step 7:
    for system_class in system_class_lists:
        requirements = system_class_lists[system_class]  
            
        # Step 8: Computing similarities in batches
        for i in range(0, len(requirements), batch_size):
            
            # Step 9: Extracting a batch of requirements for similarity calculation
            batch_reqs = requirements[i:i+batch_size]
            batch_vectors = [sentence_to_vector(req, model) for req in batch_reqs]
            
            # Step 10: Calculating cosine similarities between batch vectors and training vectors
            similarities = np.dot(batch_vectors, np.transpose(train_vectors))
            similarities /= np.outer(np.linalg.norm(batch_vectors, axis=1), np.linalg.norm(train_vectors, axis=1))
            
            # Step 11: Finding the indices of the two closest requirements for each batch requirement
            closest_indices = np.argsort(similarities, axis=1)[:, -2:]
            closest_reqs = [req for indices in closest_indices for req in [dataset[idx] for idx in indices]]
            
            # Step 12: Adding the closest requirements to the new_requirements list
            new_requirements.extend(closest_reqs)
            
        # Step 13: Removing duplicate or identical requirements from the new_requirements list
        new_requirements = list(set(new_requirements))   

        # Step 14: Classify each requirement from the new data
        frnfr_categories, system_classes = [], []
        frnfr_categories, system_classes = Classification_module(new_requirements, frnfr_categories, system_classes)

        # Step 15: Filter new requirements by system class
        filtered_requirements, filtered_frnfr_categories, filtered_system_classes = [], [], []

        for requirement, frnfr_category, new_system_class in zip(new_requirements, frnfr_categories, system_classes):
            if new_system_class == system_class:
                filtered_requirements.append(requirement)
                filtered_frnfr_categories.append(frnfr_category)
                filtered_system_classes.append(new_system_class)        

        # Step 16: Add the filtered new requirements, FR/NFR categories, and system classes to the delta library
        delta_list.setdefault(system_class, []).extend(list(zip(filtered_requirements, filtered_frnfr_categories, filtered_system_classes)))

    # Step 16: Return the delta_lists to the main script
    return delta_list


##################################################################################################################
def delta_lists_creation_module(delta_list):
    """
    Create a Delta List of relevant requirements, that should be added
    to complete the project, from the delta_list for each system_class.
    """

    # Step 1: Define the name of the output directory for the Delta Lists
    delta_list_dir = "Delta_Lists"

    # Step 2: Create the output directory for the Delta Lists if it doesn't exist
    os.makedirs(delta_list_dir) if not os.path.exists(delta_list_dir) else None

    # Step 3: Create a Delta List for each system_class in the delta_list dictionary
    for system_class in delta_list:
        # Step 4: Create the output file name for the Delta List based on the system class
        output_file = os.path.join(delta_list_dir, f"{system_class}_Delta_List.csv")

        # Step 5: Generate a list of requirement numbers
        requirements_numbers = list(range(1, len(delta_list[system_class]) + 1))

        # Step 6: Write the Delta List to a CSV file
        delta_df = pd.DataFrame({
            "Requirement Number": requirements_numbers,
            "Requirement": [x[0] for x in delta_list[system_class]],
            "FR/NFR Category": [x[1] for x in delta_list[system_class]],
            "System Class": [x[2] for x in delta_list[system_class]]
        })
        delta_df.to_csv(output_file, index=False)

##################################################################################################################
if __name__ == '__main__':

    # Step 1: Upload the file
    file = 'NLP4ReF-NLTK/Initial_Files/Thesis_Req_Test.csv'
    requirements_numbers, requirements, frnfr_categories, system_classes = upload_module(file)

    # Step 2: Normalize the requirements
    filtered_requirements = normalization_module(requirements)

    # Step 3: Classify the data
    frnfr_categories, system_classes = Classification_module(requirements, frnfr_categories, system_classes)

    # Step 4: Vectorize the filtered requirements
    new_file_module(requirements_numbers, requirements, system_classes, frnfr_categories)

    # Step 5: Create a library of all the requirements within there systems class
    system_class_lists = sys_list_arrangement(system_classes, requirements)
    
    # Step 6: Forecaste the new requirements base on the  System class library
    delta_list = Forecaste_module(system_class_lists)

    # Step 7: Create Delta Lists for each system class in the Delta List
    delta_lists_creation_module(delta_list)
    
##################################################################################################################