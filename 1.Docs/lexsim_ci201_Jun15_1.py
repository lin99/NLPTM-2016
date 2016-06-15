# -*- coding: cp1252 -*-
from scipy.stats.stats import pearsonr
from nltk.stem.porter import *


# funcion que lee un dataset de similiud lexica
def leer_dataset(nombre_archivo):
    dataset={}
    source_file=open(nombre_archivo,"r")
    for linea in source_file.readlines():
        posicion_primer_tab=linea.find("\t")
        posicion_segundo_tab=linea.find("\t",posicion_primer_tab+1)
        
                                       
        palabra1=linea[0:posicion_primer_tab]
        palabra2=linea[posicion_primer_tab+1:posicion_segundo_tab]
        gold_standard=linea[posicion_segundo_tab+1:-1]
        gold_standard=float(gold_standard)

        dataset[(palabra1,palabra2)]=gold_standard
    return dataset



####################################################################
def lex_sim_jaccard(palabra1,palabra2):
    A=set(palabra1)
    B=set(palabra2)
    AuB=A.union(B)
    AiB=A.intersection(B)
    similitud=float(len(AiB))/float(len(AuB))
    return similitud

def lex_sim_cosine(palabra1,palabra2):
    A=set(palabra1)
    B=set(palabra2)
    AuB=A.union(B)
    AiB=A.intersection(B)
    similitud=float(len(AiB))/(len(A)*len(B)**0.5)
    return similitud

def lex_sim_Jaro(word1,word2):
        
    # determines the longer and shorter strings
    if len(word1)>len(word2):
        long_word=word1
        short_word=word2
    else:
        long_word=word2
        short_word=word1
    # establish the window size
    window=len(long_word)/2-1
    if window<0:
        window=0
    # initializes list of matching positions
    long_matches=[]
    short_matches=[]
    tlong_word=long_word
    for i in range(0,len(short_word)):
        lower_bound=max([0,i-window])
        upper_bound=min([len(tlong_word)-1,i+window])
        j=tlong_word.find(short_word[i],lower_bound,upper_bound+1)
        if j>=0: # sucsessfull match
            tlong_word=tlong_word[0:j]+'#'+tlong_word[j+1:]  # the character can't be matched again!
            long_matches.append(j)
            short_matches.append(i)
    matches=len(long_matches)  # or len(short_matches
    long_matches.sort()
    transpositions=0
    for i in range(0,matches):
        if not(short_word[short_matches[i]]==long_word[long_matches[i]]):
            transpositions=transpositions+1
    transpositions=1.0*transpositions/2
    if matches==0:
            return 0.0
    return ((1.0*matches/len(short_word))+(1.0*matches/len(long_word))+((0.0+matches-transpositions)/matches))/3.0



# Edit distance
#
# Levenshtein VI (1966). "Binary codes capable of correcting deletions, insertions, and reversals". Soviet Physics Doklady 10: 707–10.
# (http://en.wikipedia.org/wiki/Edit_distance)
# R.A. Wagner and M.J. Fischer. 1974. The String-to-String Correction Problem. Journal of the ACM, 21(1):168–173.
def edit_distance(word1,word2):
    d0=range(0,len(word2)+1)
    d1=range(0,len(word2)+1)
    for i in range(1,len(word1)+1):
        d1[0]=i
        for j in range(1,len(word2)+1):
            deletion_cost=1
            insertion_cost=1
            if word1[i-1]==word2[j-1]:
                substitution_cost=0
            else:
                substitution_cost=1
            d1[j]=min([d1[j-1]+insertion_cost,d0[j]+deletion_cost,d0[j-1]+substitution_cost])
        for k in range(0,len(word2)+1):
            d0[k]=d1[k]
    max_distance=max([len(word1),len(word2)])
    min_distance=0
    return d1[len(word2)]

def lex_sim_edit_distance(word1,word2):
    return 1-float(edit_distance(word1,word2))/max([len(word1),len(word2)])
####################################################################


#print lex_sim_Jaro("Gonzalo","Gonzalez")
#print lex_sim_Jaro("Perez","Gonzalez")

print lex_sim_edit_distance("Gonzalo","Gonzalez")
print lex_sim_edit_distance("Perez","Gonzalez")


#exit()

nombres_datasets=["MC","MEN","MTURK287","MTURK771","REL122","RG","RW","SCWS","SL999","VERB143","WS353","WSR","WSS","YP130"]

stemmer = PorterStemmer()

print "Dataset\t#pares\t(Pearson_r,p-value)"
for nombre_dataset in nombres_datasets:
    dataset=leer_dataset("./data_lexsim/"+nombre_dataset+".txt")
    predicciones=[]
    gold_standard=[]
    for palabra1,palabra2 in dataset:
        _palabra1=stemmer.stem(palabra1)
        _palabra2=stemmer.stem(palabra2)
        #prediccion=lex_sim_cosine(palabra1,palabra2)
        prediccion=lex_sim_jaccard(_palabra1,_palabra2)
        #prediccion=lex_sim_Jaro(palabra1,palabra2)
        #prediccion=lex_sim_edit_distance(_palabra1,_palabra2)
        predicciones+=[prediccion]
        GS=dataset[(palabra1,palabra2)]
        gold_standard+=[GS]
    print nombre_dataset,"\t",len(dataset),"\t",pearsonr(gold_standard,predicciones)[0]










#print len(dataset1),dataset1

#dataset2=leer_dataset("./en/RG.txt")
#print len(dataset2),dataset2





