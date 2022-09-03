from sklearn import metrics as mtr
from math import *
from scipy.stats import norm
import numpy.polynomial.polynomial as poly
import numpy
import subprocess
import matplotlib.pyplot as plt
import numpy as np

tablica = numpy.arange(18, 163, 8)
print(range(len(tablica)))
il_prob = 100
dane = [] * (len(tablica))
powt = [] * il_prob

for i in tablica:
    zmienna = subprocess.check_output(['z10v21.exe', str(i)])
    dane.append(zmienna)
for q in range(len(dane)):
    dane[q] = float(float(dane[q]))

mind = float(min(dane))
maxd = float(max(dane))
print("Maksymalna wartosc listy to: " + str(maxd) + "   Minimalna wartosc to: " + str(mind) + "\n")

print("Pierwszy parametr wynosi: " + str(float(dane[0])) + "\nOstatni parametr wynosi: " + str(
    float(dane[len(dane[:0:-1])])) + "\n")
il_prob = 100


def wielkosc_proby(a, parametr):
    for z in range(a):
        wielkp = subprocess.check_output(['z10v21.exe', str(parametr)])
        powt.append(wielkp)
    for q in range(len(powt)):
        powt[q] = float(float(powt[q]))


wielkosc_proby(il_prob, 100)

print(powt)
for h in range(len(powt)):
    powt[h] = float(float(powt[h]))

n = len(powt)
# srednia
j = 0
f = 0
for gyr in range(n):
    gyr = float(powt[gyr])
    f = f + gyr

srednia = float(f / n)
print("srednia wynosi: " + str(srednia) + "\n")


###
# Mediana

def mediana(lista):
    t = len(lista)
    lista.sort()

    if t % 2 == 0:
        median1 = lista[t // 2]
        median2 = lista[t // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = lista[t // 2]
    print("Mediana wynosi: " + str(median) + "\n")


mediana(powt)

###

# Wariancja

sum2 = 0
for i in range(n):
    sum2 = sum2 + (powt[i] - srednia) ** 2

myvar1 = sum2 / n

print("Wariancja: ", myvar1)
print("\n")

# print(len(l))

# odchylenie standardowe


variance = sum([((x - srednia) ** 2) for x in powt]) / n
res = variance ** 0.5
print("Odchylenie standardowe: " + str(res) + "\n")

# Max i min w probie


print("Maksymalna wartosc proby to: " + str(max(powt)) + "   Minimalna wartosc proby to: " + str(min(powt)))
print("\n")

onesx = np.ones(il_prob) * 100

plt.plot(onesx, powt, '.', color="blue")
plt.show()

bins = 'auto'
mu, std = norm.fit(powt)

plt.hist(powt, bins='auto', density=True, alpha=0.6, color='red')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

plt.plot(tablica, dane, 'o', color="red")
plt.xlabel('Roznica naprezen ∆σ')
plt.ylabel('Liczba cykli Nk')
plt.show()
# standaryzacja
linia = 2. * (tablica - np.min(tablica)) / np.ptp(tablica) - 1
plt.plot(linia, dane, 'o', color="green")
plt.ylabel('Liczba cykli Nk')
plt.show()


def aproksymacja_wielomianowa(x, y, stp):
    model = np.linspace(x[0], x[-1], num=len(x))
    wspolcz = poly.polyfit(x, y, stp)
    fit_funk = poly.polyval(model, wspolcz)
    print(f"Wspolczynniki: {stp} i {wspolcz}")
    print("\n")

    plt.scatter(x, y, color="green")
    plt.plot(model, fit_funk)
    plt.title(f"Wielomian stopnia {stp}")
    plt.xlabel('')
    plt.ylabel('Liczba cykli')
    plt.show()

    bl = []
    bl1 = []
    l_apro = range(19)
    for licznik in l_apro:
        bl.append(fabs(y[licznik] - fit_funk[licznik]))

    print(f"Maksymalna wartosc bledu bezwzglednego wielomianu stopnia {stp}:{max(bl)}")
    print("\n")
    print(f"srednia wartosc bledu bezwzglednego wielomianu stopnia {stp}:{np.average(bl)}")
    print("\n")

    for licznik2 in l_apro:
        bl1.append(fabs(y[licznik2] - fit_funk[licznik2]) / fit_funk[licznik2])
    print(f"Maksymalna wartosc bledu wzglednego wielomianu stopnia {stp}:{max(bl1)}")
    print("\n")
    print(f"srednia wartosc bledu wzglednego wielomianu stopnia {stp}:{np.average(bl1)}")

    r2 = mtr.r2_score(y, fit_funk)
    print(f'R^2=wielomianu stopnie {stp}::{r2}')
    print("\n")
    RSME = mtr.mean_squared_error(y, fit_funk)

    print(f'RSME=wielomianu stopnia {stp}::{RSME}')


for i in range(1, 20):
    aproksymacja_wielomianowa(tablica, dane, i)


def aproksymacja_logarytmiczna(x, y, stp):
    model = np.linspace(x[0], x[-1], num=len(x))
    wspolcz = poly.polyfit(x, y, stp)
    fit_funk = poly.polyval(model, wspolcz)
    print(f"Wspolczynniki: {stp} i {wspolcz}")
    print("\n")

    plt.scatter(x, y, color="red")
    plt.plot(model, fit_funk)

    plt.xlabel('')
    plt.ylabel('Liczba cykli')
    plt.show()

    bl = []
    bl1 = []
    l_apro = range(19)
    for licznik in l_apro:
        bl.append(fabs(y[licznik] - fit_funk[licznik]))

    print(f"Maksymalna wartosc bledu bezwzglednego funkcji logarytmicznej stopnia  {stp}:{max(bl)}")
    print("\n")
    print(f"Srednia wartosc bledu bezwzglednego funkcji logarytmicznej stopnia  {stp}:{np.average(bl)}")
    print("\n")

    for licznik2 in l_apro:
        bl1.append(fabs(y[licznik2] - fit_funk[licznik2]) / fit_funk[licznik2])
    print(f"Maksymalna wartosc bledu wzglednego funkcji logarytmicznej stopnia  {stp}:{max(bl1)}")
    print("\n")
    print(f"Srednia wartosc bledu wzglednego funkcji logarytmicznej stopnia  {stp}:{np.average(bl1)}")

    r2 = mtr.r2_score(y, fit_funk)
    print(f'R^2=logarytmu stopnie {stp}::{r2}')
    print("\n")
    RSME = mtr.mean_squared_error(y, fit_funk)

    print(f'RSME=logarytmu stopnie  {stp}::{RSME}')


zlog = [log(i) for i in tablica]
print(zlog)
for i in range(1, 20):
    aproksymacja_logarytmiczna(zlog, dane, i)
