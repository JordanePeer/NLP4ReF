import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#########################################################################################################################################################
FR_NFR_classes = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
Sys_classes = ['AC', 'SE', 'DS', 'SW', 'UI', 'CO']

def get_counts(df, class_col_name, classes):
    counts = {key:0 for key in classes}
    for key in counts.keys():
        counts[key] = df[df[class_col_name] == key].shape[0]
    return counts

# read datasets and count
nfr_df = pd.read_csv('NLP4ReF-NLTK/Initial_Files/Thesis_Req_Test.csv', sep=',', header=0, quotechar = '"')
initial_nfr_counts = get_counts(nfr_df, 'FR/NFR Category', FR_NFR_classes)
initial_sys_counts = get_counts(nfr_df, 'System Class', Sys_classes)


exp_df = pd.read_csv('NLP4ReF-NLTK/Output_Files/Thesis_Req_Test/Thesis_Req_Output.csv', sep=',', header=0, quotechar = '"', doublequote=True)
output_nfr_counts = get_counts(exp_df, 'FR/NFR Category', FR_NFR_classes)
output_sys_counts = get_counts(exp_df, 'System Class', Sys_classes)



#########################################################################################################################################################
# calculate values for plot for all classes
initial_values = [initial_nfr_counts[key] for key in FR_NFR_classes]
output_values = [output_nfr_counts[key] for key in FR_NFR_classes]
indeces = np.arange(len(FR_NFR_classes))
bar_width = 0.45

fig, ax = plt.subplots()
ax.bar(indeces, initial_values, bar_width, label='Initial')
ax.bar(indeces + bar_width, output_values, bar_width, label='Output')

# draw plot for all classes
ax.set_title('12 FR/NFR class distribution')
ax.set_xlabel('Requirement type')
ax.set_ylabel('Count')
ax.set_xticks(indeces + bar_width / 2)
ax.set_xticklabels(FR_NFR_classes)
ax.set_yticks(np.arange(0, 29, 1))
ax.legend()

plt.savefig('FR_NFR_plot_classes.png')
plt.close()



#########################################################################################################################################################
# calculate values for plot for all classes
nfr_values = [initial_sys_counts[key] for key in Sys_classes]
output_values = [output_sys_counts[key] for key in Sys_classes]
indeces = np.arange(len(Sys_classes))
bar_width = 0.45

fig, ax = plt.subplots()
ax.bar(indeces, nfr_values, bar_width, label='Initial')
ax.bar(indeces + bar_width, output_values, bar_width, label='Output')

# draw plot for all classes
ax.set_title('6 System class distribution')
ax.set_xlabel('Requirement type')
ax.set_ylabel('Count')
ax.set_xticks(indeces + bar_width / 2)
ax.set_xticklabels(Sys_classes)
ax.set_yticks(np.arange(0, 50, 2))
ax.legend()

plt.savefig('Systems_Class_plot_classes.png')
plt.close()


#########################################################################################################################################################
# calculate values for plot for FR vs NFR
initial_count_nfr = sum([initial_nfr_counts[key] for key in FR_NFR_classes[1:]])
initial_values = [initial_nfr_counts['F'], initial_count_nfr]
output_count_nfr = sum([output_nfr_counts[key] for key in FR_NFR_classes[1:]])
exp_diffs = [output_nfr_counts['F'], output_count_nfr]
indeces = np.arange(2)
bar_width = 0.45

fig, ax = plt.subplots()
ax.bar(indeces, initial_values, bar_width, label='Initial')
ax.bar(indeces + bar_width, exp_diffs, bar_width, label='Output')

# draw plot for FR vs NFR
ax.set_title('FR versus NFR class distribution')
ax.set_xlabel('Requirement type')
ax.set_ylabel('Count')
ax.set_xticks(indeces + bar_width / 2)
ax.set_xticklabels(['FR', 'NFR'])
ax.set_yticks(np.arange(0, 55, 2))
ax.legend()

fig.set_size_inches(4.5, 4.8)
fig.savefig('stacked_plot_fr_nfr.png')