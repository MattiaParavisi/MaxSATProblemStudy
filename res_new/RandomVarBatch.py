import random as rnd
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as st
import os

clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

#inizializza una classe per generare le CNF dato il numero di variabili x0 .. xn che si potranno utilizzare
#se come parametro si passa 50 allora le variabili che saranno presenti nelle formule saranno x0 .. x49
class gen_clause:
    
    #Costruttore: inizializza una variabile interna al numero di var - 1 perché la libreria random di python considera gli estremi inclusi nella
    #generazione di nuovi numeri
    def __init__(self, num_of_var, prob_not = 0.5):
        self.num_of_var = num_of_var - 1
        self.prob_not = prob_not
    
    #Funzione: genera un numero tot_clause_num di clausole con AL PIU' num_of_var_for_clause di variabili per ogni singola clausola
    #Restituisce: Una lista di liste in cui ogni sottolista è una clausola
    def gen_CNF(self, tot_clause_num, num_of_var_for_clause):
        final_CNF = []
        for i in range (tot_clause_num):
            final_CNF.append(self.__gen_OR(num_of_var_for_clause, self.prob_not))
        return final_CNF
    
    #Funzione privata: utilizzata da gen_CNF per generare una generica clausola con al più num_of_var_for_clause variabili in ogni clausola
    #La clausola è con le variabili in ordine decrescente di indici per favorire le successive sostituzioni
    #Restituisce: una clasuola con al più num_of_var_for_clause variabili ordinate in ordine decrescente secondo gli indici
    def __gen_OR(self, num_of_var_for_clause, prob_not):
        ret_OR = ''
        index_vec = sorted(rnd.sample(range(0, self.num_of_var), num_of_var_for_clause), reverse = True)
        num_vars = rnd.randint(1, num_of_var_for_clause)
        for i in range (num_vars):
            prob = rnd.random()
            if(i == num_vars - 1):
                if(prob < prob_not):
                    ret_OR += 'x' + str(index_vec[i])
                else:
                    ret_OR += 'not x' + str(index_vec[i])
                break
            if(prob < prob_not):
                ret_OR += 'not x' + str(index_vec[i]) + ' or '
            else:
                ret_OR += 'x' + str(index_vec[i]) + ' or '
        return ret_OR

#Funzione obiettivo: presa una CNF e il numero di variabili n_bits che possono essere generate globalmente per quella CNF e un vettore di assegnamenti random per le variabili
#Restituisce: il valore della funzione obiettivo per quella CNF, quindi restituisce il numero di clausole vere. 
def objective(CNF, n_bits, pop):
    #Inizializza un dizionario vuoto
    dict_value = {}
    #Copia la CNF per non modificarla globalmente
    CNF_in = CNF.copy()
    #Per ogni variabile possibile crea una coppia nel dizionario che corrisponde all'assegnamento random che è stato scelto per la varibile i-esima
    #In questo caso non abbiamo paura dei duplicati perché ogni variabile compare X volte nella corrente CNF ma con 1 solo assegnamento, quindi basta 1 coppia chiave-valore
    for i in range(n_bits):
        dict_value['x'+str(i)] = pop[i]
    #Utilizzando un'espressione regolare si estraggono gli indici delle variabili presenti nella clausola per ogni clausola nella CNF che sarannno in ordine decrescente    
    for i in range(len(CNF_in)):
        index_cnd_i = re.findall(r'\d+', CNF[i])
        #Per ogni indice si sostituisce la variabile con il valore nel dizionario
        for j in index_cnd_i:
            CNF_in[i] = CNF_in[i].replace('x' + str(j), str(dict_value['x' + str(j)]))
    #Viene valutata ogni clausola nella CNF (che può essere true o false o 0 o 1) e sommati i risultati (True = 1, False = 0, 1 = 1, 0 = 0)        
    return sum([eval(CNF_in[i]) for i in range(len(CNF_in))])

#Facciamo 200 prove di assegnamenti random per ogni CNF
num_rep = 5000
#La CNF globalmente avrà variabili del tipo x0..x14
n_bits = 15
#Dichiaro la prob di avere un not
prob_not = 25

#Le prove sono fatte per 10 CNF differenti
for outer_i in range(1):
    #Vogliamo che globalmente la variabili siano da 0 a 14 quindi inzializziamo il generatore con 15
    my_gen_clause = gen_clause(n_bits, prob_not)
    #Vogliamo 2000 clausole con al più 5 variabili ognuna
    CNF = my_gen_clause.gen_CNF(2000, 5)

    #print(CNF)

    #Creo una lista di risultati per generare i grafici e impongo il best a 0
    lst_res = []

    best_res = 0
    
    #Per 200 volte genero assegnamenti random per le 15 variabili (da 0 a 14)
    for _ in tqdm (range (num_rep), desc="Eseguo la verifica delle clausole", ascii=False, ncols=75):
        pop = np.random.randint(0, 2, n_bits).tolist()
        #Calcolo la funzione obiettivo per quella CNF
        res = objective(CNF, n_bits, pop)
        #Appendo risulalto corrente
        lst_res.append(res)
        #Aggiorno il migliore risultato ottenuto
        if(res > best_res):
            best_res = res
            print('Best result: ', res)
            clearConsole()
    
    #Ho notato che i risultati precedenti assomigliavano ad una distribuzione normale
    #Calcolo gli stimatori non deviati per mu e sigma che sono la media e la dev-std
    media = np.array(lst_res).mean()
    std = np.array(lst_res).std()
    print(media, std)
    
    #Creo una var aleatoria con la distribuzione normale di parametri mu e sigma stimati
    mynorm = st.norm(loc = media, scale = std)
    
    #Plotto tra media - 3std e media + 3std per avere la visione centralizzata della campana gaussiana
    x = np.linspace(media - 3*std, media + 3*std)

    #Calcolo le y corrispondenti alle x calcolate precedentemente
    y = mynorm.pdf(x)

    #Le y tra l'istogramma dei res e della norm sono diverse, quindi mi serve un fattore correttivo da applicare alla normale per mantenere i risultati dell'istogramma
    #Prendo il massimo valore delle y dell'istogramma e voglio che y.max() * fatt_correttivo = ymaxistogramma.max(), quindi calcolo il fattore correttivo e lo applico alle y della normale
    #Plotto i risultati
    ymax, _, _ = plt.hist(lst_res, alpha = 0.8)
    corrective_factor = ymax.max() / max(y)
    plt.plot(x, y * corrective_factor)
    plt.xlabel('Obj. function')
    plt.legend()
    plt.savefig('graph'+str(outer_i)+'.png')
    plt.clf()
