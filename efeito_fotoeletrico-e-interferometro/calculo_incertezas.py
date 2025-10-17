"""
Cálculo das incertezas na regressão linear e na constante de Planck
Experimento: Efeito Fotoelétrico
"""

import numpy as np
import pandas as pd
from scipy import stats

# Constantes
e = 1.602e-19  # Carga do elétron em Coulombs

# Ler dados experimentais
df = pd.read_csv('dados_variacao_intensidade.csv')

# Extrair frequências e potenciais de freamento
frequencias = df['freq'].values * 1e14  # Hz
V0_8mm = df['V0_8mm'].values  # Volts

# Cálculo 1: Regressão para abertura de 8mm
print("=" * 60)
print("REGRESSÃO LINEAR - ABERTURA 8mm")
print("=" * 60)

# Regressão linear: V0 = a*nu + b
slope_8mm, intercept_8mm, r_value, p_value, std_err = stats.linregress(frequencias, V0_8mm)

print(f"\nCoeficiente angular (a): {slope_8mm:.3e} V·s")
print(f"Coeficiente linear (b): {intercept_8mm:.3f} V")
print(f"R²: {r_value**2:.6f}")

# Cálculo da incerteza no coeficiente angular (sigma_a)
# Fórmula: sigma_a = sqrt[(1/(N-2)) * sum(residuals^2) / sum((xi - x_mean)^2)]

N = len(frequencias)
y_pred = slope_8mm * frequencias + intercept_8mm
residuals = V0_8mm - y_pred
sum_residuals_sq = np.sum(residuals**2)
x_mean = np.mean(frequencias)
sum_x_deviation_sq = np.sum((frequencias - x_mean)**2)

sigma_a_8mm = np.sqrt((1/(N-2)) * sum_residuals_sq / sum_x_deviation_sq)

print(f"\nCálculo de sigma_a:")
print(f"  N = {N}")
print(f"  Soma dos resíduos² = {sum_residuals_sq:.6e}")
print(f"  Soma de (xi - x_mean)² = {sum_x_deviation_sq:.6e}")
print(f"  sigma_a = {sigma_a_8mm:.3e} V·s")

# Cálculo da incerteza na constante de Planck (sigma_h)
# Fórmula: sigma_h = e * sigma_a

h_8mm = e * slope_8mm
sigma_h_8mm = e * sigma_a_8mm

print(f"\nConstante de Planck:")
print(f"  h = e × a = {h_8mm:.3e} J·s")
print(f"  sigma_h = e × sigma_a = {sigma_h_8mm:.2e} J·s")

# Valor aceito
h_aceito = 6.626e-34
erro_percentual = abs(h_8mm - h_aceito) / h_aceito * 100

print(f"\nComparação com valor aceito:")
print(f"  h_aceito = {h_aceito:.3e} J·s")
print(f"  Erro percentual = {erro_percentual:.2f}%")

# Cálculo 2: Média de todas as aberturas
print("\n" + "=" * 60)
print("REGRESSÃO LINEAR - MÉDIA DE TODAS AS ABERTURAS")
print("=" * 60)

# Calcular média dos V0 para todas as aberturas
V0_media = df[['V0_2mm', 'V0_4mm', 'V0_8mm']].mean(axis=1).values

# Regressão linear com média
slope_media, intercept_media, r_value_media, p_value_media, std_err_media = stats.linregress(frequencias, V0_media)

print(f"\nCoeficiente angular (a): {slope_media:.3e} V·s")
print(f"Coeficiente linear (b): {intercept_media:.3f} V")
print(f"R²: {r_value_media**2:.6f}")

# Cálculo de sigma_a para média
y_pred_media = slope_media * frequencias + intercept_media
residuals_media = V0_media - y_pred_media
sum_residuals_sq_media = np.sum(residuals_media**2)

sigma_a_media = np.sqrt((1/(N-2)) * sum_residuals_sq_media / sum_x_deviation_sq)

print(f"\nCálculo de sigma_a:")
print(f"  sigma_a = {sigma_a_media:.3e} V·s")

# Cálculo de sigma_h para média
h_media = e * slope_media
sigma_h_media = e * sigma_a_media

print(f"\nConstante de Planck:")
print(f"  h = e × a = {h_media:.3e} J·s")
print(f"  sigma_h = e × sigma_a = {sigma_h_media:.2e} J·s")

erro_percentual_media = abs(h_media - h_aceito) / h_aceito * 100

print(f"\nComparação com valor aceito:")
print(f"  h_aceito = {h_aceito:.3e} J·s")
print(f"  Erro percentual = {erro_percentual_media:.2f}%")

# Resumo final
print("\n" + "=" * 60)
print("RESUMO - VALORES PARA O RELATÓRIO")
print("=" * 60)

print(f"\nABERTURA 8mm:")
print(f"  a = ({slope_8mm/1e-15:.3f} ± {sigma_a_8mm/1e-15:.3f}) × 10^-15 V·s")
print(f"  h = ({h_8mm/1e-34:.3f} ± {sigma_h_8mm/1e-34:.2f}) × 10^-34 J·s")
print(f"  Erro: {erro_percentual:.1f}%")

print(f"\nMÉDIA DAS ABERTURAS:")
print(f"  a = ({slope_media/1e-15:.3f} ± {sigma_a_media/1e-15:.3f}) × 10^-15 V·s")
print(f"  h = ({h_media/1e-34:.3f} ± {sigma_h_media/1e-34:.2f}) × 10^-34 J·s")
print(f"  Erro: {erro_percentual_media:.1f}%")

print("\n" + "=" * 60)
