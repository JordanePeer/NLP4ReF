
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu




##################################################################################################################
def upload_module(file):

    # Step 1: Read the CSV file and Skip the header row
    df = pd.read_csv(file, sep=',', header=0, quotechar = '"', doublequote=True)

    # Step 2: Initialize empty lists for each column
    frnfr_categories, system_classes, requirements = [], [], []

    # Step 3: Iterate through each row of the CSV file
    for i, row in df.iterrows():
        # Step 4: Append the value of each column to the corresponding list
        requirements.append(row['Requirement'])    
        frnfr_categories.append(row['FR/NFR Category'])
        system_classes.append(row['System Class'])

    # Step 7: return the lists to the main script
    return(frnfr_categories, system_classes, requirements)



###################################################################################################
def evaluate_metrics(y_test, y_pred):

    metrics_prf = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    return [metrics_prf[0], metrics_prf[1], metrics_prf[2]]

##################################################################################################################
def verify_authenticity(new_requirements, existing_requirements):
    # Verify authenticity of new requirement
    similar_count = 0
    authenticity_percentage = 0
    total_count = len(existing_requirements)
    total_new = len(new_requirements)

    for new_req in new_requirements:
        for req in existing_requirements:
            similarity = SequenceMatcher(None, new_req, req).ratio()
            if similarity > 0.8:
                similar_count += 1
                
        # Calculate authenticity percentage for each new requirement
        authenticity_percentage += (total_count - similar_count) / total_count
        # Reset similar_count for the next new requirement
        similar_count = 0  

    authenticity_percentage = authenticity_percentage / total_new * 100
    return authenticity_percentage


##################################################################################################################
def verify_relative_accuracy(new_requirements, existing_requirements):
    # Verify relative accuracy of new requirement
    vectorizer = TfidfVectorizer()
    similarity_sum = 0
    total_new = len(new_requirements)

    for new_req in new_requirements:
        max_similarity = 0
        for req in existing_requirements:
            requirements = [req, new_req]
            tfidf_matrix = vectorizer.fit_transform(requirements)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity = similarity_matrix[0][0]
            max_similarity = max(max_similarity, similarity)
        similarity_sum += max_similarity

    relative_accuracy_percentage = (similarity_sum / total_new) * 100
    return relative_accuracy_percentage




##################################################################################################################
def calculate_bleu_score(new_requirements, existing_requirements):
    # Calculate BLEU score
    total_new = len(existing_requirements)
    bleu_score = 0
    
    for new_req in new_requirements:
        new_req = new_req.split()
        req_scores = []
        for req in existing_requirements:
            req = req.split()
            req_scores.append(sentence_bleu([new_req], req))
        bleu_score += max(req_scores)

            
    bleu_score = bleu_score / total_new * 100        
    return bleu_score

##################################################################################################################
def calculate_self_bleu_score(new_requirements):
    # Calculate self-BLEU score
    total_bleu_score = 0.0
    total_new = len(new_requirements)

    for i, req in enumerate(new_requirements):
        other_reqs = new_requirements[:i] + new_requirements[i+1:]
        candidate = req.split()  
        bleu_scores = []
        for other_req in other_reqs:
            other_req = other_req.split()
            bleu_scores.append(sentence_bleu([other_req], candidate))

        if len(bleu_scores) > 0:
            total_bleu_score += max(bleu_scores)

    self_bleu_score = total_bleu_score / total_new * 100
    return self_bleu_score

##################################################################################################################
# Step 1: Upload the Initial file
file = "NLP4ReF-GPT/Initial_Files/Tram_IoT_requirements.csv"
frnfr_initial, system_initial, req_initial = upload_module(file)
    
# Step 2: Upload the New file
file = "NLP4ReF-GPT/Output_Files/Tram_IoT_new_requirements.csv"
frnfr_output, system_output, req_output = upload_module(file)

# Step 3: test Metrics 1 - 3 for FR/NFR
Test_result = evaluate_metrics(frnfr_initial, frnfr_output)
line = "FR/NFR: precision = {}; recall = {}; f1_score = {}".format(Test_result[0], Test_result[1], Test_result[2])
print(line)

# Step 4: test Metrics 1 - 3 for Systems
Test_result = evaluate_metrics(system_initial, system_output)
line = "Systems: precision = {}; recall = {}; f1_score = {}".format(Test_result[0], Test_result[1], Test_result[2])
print(line)

# Step 5: test Metrics 4 - 7 for New requirements
authenticity_percentage = verify_authenticity(req_output, req_initial)
print("Authenticity Percentage:", authenticity_percentage)

relative_accuracy_percentage = verify_relative_accuracy(req_output, req_initial)
print("Relative Accuracy Percentage:", relative_accuracy_percentage)

bleu_score = calculate_bleu_score(req_output, req_initial)
print("BLEU Score:", bleu_score)

self_bleu_score = calculate_self_bleu_score(req_output)
print("Self-BLEU Score:", self_bleu_score)

