# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:26:24 2020

@author: Giovanni Guarnieri Soares
"""

import matplotlib.pyplot as plt
from scipy.stats import norm, genextreme
import statsfuncsprova as statsfuncs
import mfdfaprova as mfdfa
import numpy as np
from matplotlib.patches import Polygon


def cullenfrey(xd, yd, legend, title):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()
    maior = max(xd)
    polyX1 = maior if maior > 4.4 else 4.4
    polyY1 = polyX1+1
    polyY2 = 3/2.*polyX1+3
    y_lim = polyY2 if polyY2 > 10 else 10
    x = [0, polyX1, polyX1, 0]
    y = [1, polyY1, polyY2, 3]
    scale = 1
    poly = Polygon(np.c_[x, y]*scale, facecolor='#1B9AAA', edgecolor='#1B9AAA',
                   alpha=0.5)
    ax.add_patch(poly)
    ax.plot(xd, yd, marker="o", c="#e86a92", label=legend, linestyle='')
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black')
    ax.plot(0, 1.7962675925351856, label="uniform", marker='^', c='black')
    ax.plot(4, 9, label="exponential", marker='s', c='black')
    ax.plot(0, 3, label="normal", marker='*', c='black')
    ax.plot(np.arange(0, polyX1, 0.1), 3/2.*np.arange(0, polyX1, 0.1)+3,
            label="gamma", linestyle='-', c='black')
    ax.plot(np.arange(0, polyX1, 0.1), 2*np.arange(0, polyX1, 0.1)+3,
            label="lognormal", linestyle='-.', c='black')
    ax.legend()
    ax.set_ylim(y_lim, 0)
    ax.set_xlim(-0.2, polyX1)
    plt.xlabel("Skewness²")
    plt.title(title+": Cullen and Frey map")
    plt.ylabel("Kurtosis")
    plt.savefig(title+legend+"cullenfrey.png")
    plt.show()


fread = open("daily-cases-covid-19.csv", "r")
country = "South Africa,"
filename = "safrica.png"
title = "Statistcs analysis, country: {}".format(country)

y = []
date = []

# skipping lines

for line in fread:
    if "Mar 9" in line and country in line:
        break


for line in fread:
    if country in line and "excl." not in line:
        ctr, code, m, yr, data = line.split(",")
        y.append(int(data))
        date.append(m[1:])
    if country in line and "May 28" in line:
        break


x = range(len(y))

ymin = min(y)
ymax = max(y)
n = len(y)
ypoints = [(ymin + (i/n) * (ymax-ymin)) for i in range(0, n+1)]

alfa, xdfa, ydfa, reta = statsfuncs.dfa1d(y, 1)
freqs, power, xdata, ydata, amp, index, powerlaw, inicio, fim = statsfuncs.psd(y)
psi, amax, amin, a0 = mfdfa.makemfdfa(y, True)
beta = 2.*alfa-1.
print("Beta=2*Alpha-1={}".format(beta))


# Plot e ajuste do histograma da série temporal

mu, sigma = norm.fit(y)
rv_nrm = norm(loc=mu, scale=sigma)
# Estimate GEV:
gev_fit = genextreme.fit(y)
# GEV parameters from fit:
c, loc, scale = gev_fit
mean, var, skew, kurt = genextreme.stats(c, moments='mvsk')
rv_gev = genextreme(c, loc=loc, scale=scale)
# Create data from estimated GEV to plot:
gev_pdf = rv_gev.pdf(ypoints)

plt.title("PDF with data from " + country + "\nmu={0:3.5}, sigma={1:3.5}"
          .format(mu, sigma))
n, bins, patches = plt.hist(y, 60, density=1, facecolor='powderblue',
                            alpha=0.75, label="Normalized data")
plt.plot(np.arange(min(bins), max(bins)+1, (max(bins) - min(bins))/len(y)),
         gev_pdf, 'r-', lw=5, alpha=0.6, label='genextreme pdf')
plt.ylabel("Probability Density")
plt.xlabel("Value")
plt.legend()
plt.savefig("PDF"+filename)
plt.show()
plt.figure(figsize=(20, 14))

# Plot da série temporal
ax1 = plt.subplot(211)
ax1.set_title(title, fontsize=18)
ax1.plot(x, y, color="firebrick", linestyle='-', label="Data")
ax1.set_ylabel("New Cases by day")
ax1.set_xlabel("Days, starting at march 10")
# Plot e cálculo do DFA
ax2 = plt.subplot(223)
ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(alfa),
              fontsize=15)
ax2.plot(xdfa, ydfa, marker='o', linestyle='', color="#12355B", label="{0:.3}"
         .format(alfa))
ax2.plot(xdfa, reta, color="#9DACB2")
# Plot e cáculo do PSD
ax3 = plt.subplot(224)
ax3.set_title(r"Power Spectrum Density $\beta$={0:.3}".format(index),
              fontsize=15)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.plot(freqs, power, '-', color='deepskyblue', alpha=0.7)
ax3.plot(xdata, ydata, color="darkblue", alpha=0.8)
ax3.axvline(freqs[inicio], color="darkblue", linestyle='--')
ax3.axvline(freqs[fim], color="darkblue", linestyle='--')
ax3.plot(xdata, powerlaw(xdata, amp, index), color="#D65108", linestyle='-',
         linewidth=3, label='{0:.3}$'.format(index))
ax2.set_xlabel("log(s)")
ax2.set_ylabel("log F(s)")
ax3.set_xlabel("Frequência (Hz)")
ax3.set_ylabel("Potência")
ax3.legend()
plt.savefig(filename)
plt.show()

skew = statsfuncs.skewness(y)
kurt = statsfuncs.kurtosis(y)

cullenfrey([skew**2], [kurt+3], "Data", country + " Covid-19")

print("Alpha={0:.3}, 2*Alfa-1={1:.3}, Beta={2:.3}, Delta Alpha={3:.3}, \
      Alpha0={4:.3}, Aalpha={5:.3},"
      .format(alfa, 2*alfa-1, beta, (amax-amin), a0, (a0-amin)/(amax-a0)))
