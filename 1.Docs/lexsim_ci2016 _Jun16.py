# -*- coding: cp1252 -*-
from scipy.stats.stats import pearsonr
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import math

from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

stemmer = PorterStemmer()

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


# función que representa una lista como una lista de n-gramas
def n_grams(lista,n):
    ngrams=[]
    for i in range(len(lista)-n+1):
        ngrams.append(lista[i:i+n])
    return ngrams

def n_grams_spectra(lista,n_desde,n_hasta): # ojo cuidar que n_desde<n_hasta
    ngrams=[]
    for n in range(n_desde,n_hasta+1):
        ngrams+=n_grams(lista,n)
    return ngrams

#prueba de ngramas
#print n_grams_spectra("murcielago",2,3)
        
        

####################################################################
# FUNCIONES DE SIMILITUD LEXICA MORFOLOGICA
####################################################################
def lex_sim_jaccard(palabra1,palabra2):
    A=set(palabra1)
    B=set(palabra2)
    AuB=A.union(B)
    AiB=A.intersection(B)
    similitud=float(len(AiB))/float(len(AuB))
    return similitud

def lex_sim_jaccard_ngrams(palabra1,palabra2,n_desde,n_hasta):
    A=n_grams_spectra(palabra1,n_desde,n_hasta)
    B=n_grams_spectra(palabra2,n_desde,n_hasta)
    return lex_sim_jaccard(A,B)


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







####################################################################
# FUNCIONES DE SIMILITUD BASADAS EN CONOCIMIENTO (WORDNET)
####################################################################

# similitud basada en contar el mínimo numero de arcos entre pares de posibles conceptos(synsets)
def lex_sim_path(lemma1,lemma2,parametro=1):
    synsets_lemma1=wn.synsets(lemma1)
    synsets_lemma2=wn.synsets(lemma2)
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            sim=synset1.path_similarity(synset2)
            if sim>max_sim:
                max_sim=sim
    return math.pow(max_sim,float(1)/parametro)

# igual que "path" pero usando la similitud de Leacock-Chodorow
def lex_sim_lch(lemma1,lemma2):
    synsets_lemma1=wn.synsets(lemma1)
    synsets_lemma2=wn.synsets(lemma2)
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try: 
                sim=synset1.lch_similarity(synset2)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim


# Similitud de Wu-Palmer
def lex_sim_wup(lemma1,lemma2):
    synsets_lemma1=wn.synsets(lemma1)
    synsets_lemma2=wn.synsets(lemma2)
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try: 
                sim=synset1.wup_similarity(synset2)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim


# Similitud de Resnik
def lex_sim_res(lemma1,lemma2,information_content=brown_ic):
    synsets_lemma1=wn.synsets(lemma1)
    synsets_lemma2=wn.synsets(lemma2)
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try: 
                sim=synset1.res_similarity(synset2,information_content)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim

# Similitud de Jiang-Conrath
def lex_sim_jcn(lemma1,lemma2,information_content=brown_ic):
    synsets_lemma1=wn.synsets(lemma1)
    synsets_lemma2=wn.synsets(lemma2)
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try: 
                sim=synset1.jcn_similarity(synset2,information_content)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim


# Similitud de Dekan Lin
def lex_sim_lin(lemma1,lemma2,information_content=brown_ic):
    synsets_lemma1=wn.synsets(lemma1)
    synsets_lemma2=wn.synsets(lemma2)
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try: 
                sim=synset1.lin_similarity(synset2,information_content)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim



####################################################################
# FUNCIONES DE SIMILITUD HIBRIDAS
####################################################################
def lex_sim_path_edit_distance(palabra1,palabra2):
    sim=lex_sim_path(palabra1,palabra2)
    if sim==0:
        sim=lex_sim_edit_distance(palabra1,palabra2)
    return sim
    
def lex_sim_path_jaccard_23grams_porter(palabra1,palabras2):
    sim=lex_sim_path(palabra1,palabra2)
    if sim==0:
        _palabra1=stemmer.stem(palabra1)
        _palabra2=stemmer.stem(palabra2)
        sim=lex_sim_jaccard_ngrams(_palabra1,_palabra2,2,3)
    return sim



    

#print lex_sim_Jaro("Gonzalo","Gonzalez")
#print lex_sim_Jaro("Perez","Gonzalez")

print lex_sim_edit_distance("Gonzalo","Gonzalez")
print lex_sim_edit_distance("Perez","Gonzalez")


#exit()

nombres_datasets=["MC","MEN","MTURK287","MTURK771","REL122","RG","RW","SCWS","SL999","VERB143","WS353","WSR","WSS","YP130"]


embeddings={}




print "Dataset\t#pares\t(Pearson_r,p-value)"
for nombre_dataset in nombres_datasets:
    dataset=leer_dataset("./data_lexsim/"+nombre_dataset+".txt")
    predicciones=[]
    gold_standard=[]
    for palabra1,palabra2 in dataset:
        _palabra1=stemmer.stem(palabra1)
        _palabra2=stemmer.stem(palabra2)
        
        #prediccion=lex_sim_cosine(_palabra1,_palabra2)
        #prediccion=lex_sim_jaccard(_palabra1,_palabra2)
        #prediccion=lex_sim_jaccard_ngrams(_palabra1,_palabra2,2,3)
        #prediccion=lex_sim_Jaro(_palabra1,_palabra2)
        #prediccion=lex_sim_edit_distance(_palabra1,_palabra2)
        #prediccion=lex_sim_path(palabra1,palabra2,2)
        #prediccion=lex_sim_lch(palabra1,palabra2)
        #prediccion=lex_sim_wup(palabra1,palabra2)
        #prediccion=lex_sim_res(palabra1,palabra2)
        #prediccion=lex_sim_jcn(palabra1,palabra2)
        #prediccion=lex_sim_jcn(palabra1,palabra2,information_content=semcor_ic)
        #prediccion=lex_sim_lin(palabra1,palabra2)
        prediccion=lex_sim_lin(palabra1,palabra2,information_content=semcor_ic)
        
        #prediccion=lex_sim_path_edit_distance(palabra1,palabra2)
        #prediccion=lex_sim_path_jaccard_23grams_porter(palabra1,palabra2)

        predicciones+=[prediccion]
        GS=dataset[(palabra1,palabra2)]
        gold_standard+=[GS]
    print nombre_dataset,"\t",len(dataset),"\t",pearsonr(gold_standard,predicciones)[0]










#print len(dataset1),dataset1

#dataset2=leer_dataset("./en/RG.txt")
#print len(dataset2),dataset2





