from scipy.stats.stats import pearsonr

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


print lex_sim_jaccard("casa","carro")
print lex_sim_jaccard("casa","casa")
print lex_sim_jaccard("perro","gata")


nombres_datasets=["MC","MEN","MTURK287","MTURK771","REL122","RG","RW","SCWS","SL999","VERB143","WS353","WSR","WSS","YP130"]

print "Dataset\t#pares\tPearson_r"
for nombre_dataset in nombres_datasets:
    dataset=leer_dataset("./data_lexsim/"+nombre_dataset+".txt")
    predicciones=[]
    gold_standard=[]
    for palabra1,palabra2 in dataset:
        prediccion=lex_sim_cosine(palabra1,palabra2)
        predicciones+=[prediccion]
        GS=dataset[(palabra1,palabra2)]
        gold_standard+=[GS]
    print nombre_dataset,"\t",len(dataset),"\t",pearsonr(gold_standard,predicciones)[0]










#print len(dataset1),dataset1

#dataset2=leer_dataset("./en/RG.txt")
#print len(dataset2),dataset2





