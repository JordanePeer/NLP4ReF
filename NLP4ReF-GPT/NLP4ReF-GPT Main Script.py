##################################################################################################################
import csv
import os
import openai




##################################################################################################################
def upload_module(file):
    """
    Uploads the data according to a certain template.
    """
    # Step 0: Upload a CSV file
    csv_file_path = 'path/to/csv/file.csv'

    # Step 1: Read the CSV file
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Step 2: Skip the header row
        next(csv_reader)

        # Step 3: Initialize empty lists for each column
        requirements_numbers = []
        requirements = []
        frnfr_categories = []
        system_classes = []

        # Step 4: Iterate through each row of the CSV file
        for row in csv_reader:
            # Step 5: Append the value of each column to the corresponding list
            requirements_numbers.append(row[0])
            requirements.append(row[1])
            frnfr_categories.append(row[2])
            system_classes.append(row[3])

    # Step 6: return the lists to the main script
    return(requirements_numbers, requirements, frnfr_categories, system_classes)




##################################################################################################################
def categorization_module(requirements):
    """
    Create a Question for GPT API to categorizes the requirements,
    as one of the 12 FR/NFR classes.
    """

    # Step 1: Define the input question
    question = """There are 12 Functional Requirement and Non Functional Requirement Categories:
      Functional (F), Availability (A), Fault Tolerance (FT), Legal & Licensing (L), Look & Feel (LF),
        Maintainability (MN), Operability (O), Performance (PE), Portability (PO), Scalability (SC),
          Security (SE), Usability (US), Availability (A). 
          Categorize each one of the next requirements as just one of the 12 FR/NFR Categories.
          In the answer, present only the Symbol (F/A/FT/L/LF/MN/O/PE/PO/SC/SE/US) do not write the requirement!!!
          
          Answer Exanple:
          1. F
          2. SC
          
          Here is the list of Requirements:"""

    # Step 2: Add each requirement to the input question
    for requirement in requirements:
        question += f"\n{requirement}"

    # Step 3: Return the input question to the main script
    return question



##################################################################################################################
def classification_module(requirements):
    """
    Create a Question for GPT API to classifies the systems
    that the requirements represent.
    """
    # Step 1: Define the input question
    question = """There are 6 System Categories:
        Sensor (SE),  Actuator (AC), Connectivity (CO), Data Storage (DS), Software (SW), User interface (UI). 
        Categorize each one of the next requirements as just one of the 6 System Categories.
        In the answer, present only the Symbol (AC/SE/DS/SW/UI/CO) do not write the requirement!!!
        
        Answer Exanple:
        1. CO
        2. SW
                
        Here is the list of Requirements:"""

    # Step 2: Add each requirement to the input question
    for requirement in requirements:
        question += f"\n{requirement}"

    # Step 3: Return the input question to the main script
    return question

##################################################################################################################
def GPT_interface(question, api_key):
    """
    Call the GPT API and send it a question and get a response from it.
    """

    # Step 1: Set up the OpenAI API client
    openai.api_key = api_key

    # Step 2: Set up the conversation for the OpenAI API client
    conversation = []
    conversation.append({'role': 'system', 'content': question})

    # Step 2: Call the OpenAI API to categorize the requirements
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation,
    )

    # Step 7: Return the GPT response
    return response




##################################################################################################################
def reorganized_module(requirements_numbers, requirements, system_classes, frnfr_categories):
    # Step 1: Create the CSV file and write the headers
    with open("Reorganized_list.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Requirement Number", "Requirement", "FR/NFR Category", "System Class"])

        # Step 2: Write each row of data to the CSV file
        for i in range(len(requirements_numbers)):
            row = [requirements_numbers[i], requirements[i], frnfr_categories[i], system_classes[i]]
            writer.writerow(row)

    # Step 3: Return the CSV file
    return "Reorganized_list.csv"

##################################################################################################################
def req_generation_module(reorganized_list):
    """
    Generates a Delta List of relevant requirements that should be added to complete the project.
    """

    # Step 1: Define the name of the input file
    input_file = reorganized_list

    # Step 2: Create a dictionary to hold the system class lists
    system_class_lists = {}

    # Step 3: Open the input file and read each row
    with open(input_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        # Step 4: Skip the headers row
        headers = next(reader)  
        for row in reader:
            # Step 5: Extract the values for the row
            requirement_number, requirement, system_class, fr_nfr_category = row
            
            # Step 6: Check if the system class list exists in the dictionary, create it if not
            if system_class not in system_class_lists:
                system_class_lists[system_class] = []
            
            # Step 7: Add the requirement to the system class list
            system_class_lists[system_class].append(requirement)
    
    # Step 8: Return the system_class_lists to the main script
    return system_class_lists





##################################################################################################################
def delta_lists_creation_module(response, system_class, requirements):
    """
    Create a Delta List of relevant requirements, that should be added
    to complete the project, from the GPT API response.
    """

    # Step 1: Define the name of the output directory for the Delta Lists
    delta_list_dir = "Delta_Lists"

    # Step 2: Create the output directory for the Delta Lists if it doesn't exist
    if not os.path.exists(delta_list_dir):
        os.makedirs(delta_list_dir)

    # Step 3: Extract the generated Delta List from the response
    delta_list = []
    delta_list = response.choices[0].text.strip().split("\n")

    # Step 4: Create the output file name for the Delta List
    output_file = os.path.join(delta_list_dir, f"{system_class}_Delta_List.csv")

    # Step 5: Write the Delta List to a CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["System Class", "Requirement"])
        
        # Step 6: Write the requirements from the delta_list to the CSV file
        for requirement in delta_list:
            writer.writerow([system_class, requirement])
 
##################################################################################################################
def main():
    # Step 1: Upload the file
    file = "Initial_Requirements_List.csv"
    requirements_numbers, requirements, frnfr_categories, system_classes = upload_module(file)

    # Step 2: Create a question to Categorize the requirements
    question = categorization_module(requirements)

    # Step 3: Ask the GPT API to answer the question
    api_key = 'your_api_key'
    response = GPT_interface(question, api_key)

    # Step 4: Get the categorized requirements from the API response
    # and split the categorized requirements into a list and store it 
    # in the frnfr_categories list

    frnfr_categories = response.choices[0].text.strip().split("\n")

    # Step 5: Create a question to Classify the requirements
    question = classification_module(requirements)

    # Step 6: Ask the GPT API to answer the question
    response = GPT_interface(question, api_key)

    # Step 7: Get the classified requirements from the API response
    # and split the classified requirements into a list and store it 
    # in the system_classes list
    system_classes = response.choices[0].text.strip().split("\n")

    # Step 8: Reorganize the data in a CSV file
    reorganized_list = reorganized_module(requirements_numbers, requirements, frnfr_categories, system_classes)

    # Step 9: Create a question to Generate the Delta List
    system_class_lists = req_generation_module(reorganized_list)

    # Step 10: Loop through each system class
    for system_class, requirements in system_class_lists.items():
        # Step 11: Ask the OpenAI API for a Delta List of requirements
        question = f"""please generate 2 new relevant requirements for each requirement for System Class: {system_class}. 
        Here is the list of Requirements:"""
        # Step 12: Append all the requirements from the current system class to the question
        for requirement in requirements:
            question += f"\n{requirement}"
        # Step 13: Ask the GPT API to answer the question
        response = GPT_interface(question, api_key)
        # Step 14: Create a Delta List for this specific system_class
        delta_lists_creation_module(response, system_class, requirements)

    # Step 15: Present the Delta List to the Systems Engineers
    print('The delta_list for each system is under the Delta Lists Files Directory: delta_list_dir ')



##################################################################################################################
if __name__ == '__main__':
    main()
