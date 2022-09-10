#Showing the number of values in each txt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

chimp = pd.read_csv("<PATH>/chimpanzee.txt", sep="\s+")
dog = pd.read_csv("<PATH>/dog.txt", sep="\s+")
human = pd.read_csv("<PATH>/human.txt", sep="\s+")

chimp.columns = ["Sequence","Class"]
dog.columns = ["Sequence","Class"]
human.columns = ["Sequence","Class"]

chimp_counts = chimp.Class.value_counts()
dog_counts = dog.Class.value_counts()
human_counts = human.Class.value_counts()

x = np.arange(7)
width = 0.50

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, chimp_counts[x], width/2, label="Chimpanzee")
rects2 = ax.bar(x, dog_counts[x], width/2, label="Dog")
rects3 = ax.bar(x + width/2, human_counts[x], width/2, label ="Human",)


ax.set_ylabel('Number of sequences')
ax.set_title('Class distributions')
ax.set_xticks(x)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.show()

#Encoding txt files (k-mer size :6)

def encoded_df(df, size):
    enc_df = df
    for i in range(len(enc_df)):
        list_seq = []
        for j in range(len(enc_df.Sequence[i]) -size +1):        
        
            list_seq.append(enc_df.Sequence[i][j:j+size])
        
        enc_df.Sequence[i] = list_seq

    return(enc_df)

enc_chimp = encoded_df(chimp,7)
enc_dog = encoded_df(dog,7)
enc_human = encoded_df(human,7)

# Cleaning the empty line

enc_human.drop(index=[3534], axis = 0, inplace = True)
#print(enc_human)

enc_human.reset_index(drop=True, inplace=True)

# Bag of words algorithm
def bagofwords(enc_df):

    word_list = []

    BoW_array = np.zeros([len(enc_df),18915])
    for i in range(len(enc_df)):
    
        for word in enc_df.Sequence[i]:
            count = 0
            if word not in word_list:
                word_list.append(word)
                count += 1
                BoW_array[i,word_list.index(word)] = count
            else:
                count = BoW_array[i,word_list.index(word)] + 1
                BoW_array[i,word_list.index(word)] = count  

    return BoW_array

# Bag of words for all species   
#BoW_human = bagofwords(enc_human)
#BoW_chimp = bagofwords(enc_chimp)
#BoW_dog = bagofwords(enc_dog)

# Saving the Bag of words 
#np.savetxt("/content/drive/MyDrive/BoW_human.csv", BoW_human, delimiter=",")
#np.savetxt("/content/drive/MyDrive/BoW_chimp.csv", BoW_chimp, delimiter=",")
#np.savetxt("/content/drive/MyDrive/BoW_dog.csv", BoW_dog, delimiter=",")

df = pd.read_csv("/content/drive/MyDrive/BoW_human.csv", header=None)

# Creating labels
label_y = list(enc_human.Class)
human_y = np.array(label_y)
indexes = []
for i in range(len(label_y)):
    if label_y[i] == 6:
        indexes.append(i)
        i = 0
indexes = indexes[int(len(indexes)/2):]

# Deleting data to improve learning ( data with label 6)
BoW_human = df.to_numpy()
BoW_human_del = np.delete(BoW_human, indexes,0)
#print(BoW_human_del.shape)
human_y_del = np.delete(human_y, indexes, 0)
#print(human_y_del.shape)
