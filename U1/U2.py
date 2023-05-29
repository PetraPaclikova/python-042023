import pandas
from scipy import stats
data= pandas.read_csv("ukol_02_a.csv")

# res = stats.shapiro(data["97"])
# res = stats.shapiro(data["98"])

# res = stats.ttest_rel(data["97"], data["98"])
# print(res)
# print("procento lidi nespokojenych s inflaci se zmenilo, nulova hypoteza prccento lidi nespokojenych s inflaci se nezmenilo")

euro = pandas.read_csv("ukol_02_b.csv")
only_euro= pandas.read_csv("countries.csv")
df_merged= pandas.merge(euro, only_euro)

# res = stats.shapiro(df_merged["National Government Trust"])
# res = stats.shapiro(df_merged["EU Trust"])
# res = stats.pearsonr(df_merged["National Government Trust"], df_merged["EU Trust"])
# print(res)
# print(" narodostni duvera a duvera v EU spolu souvisi (nulova hypoteza- nesouvisi spolu)")
no_euro= df_merged[df_merged["Euro"]==0]
euros= df_merged[df_merged["Euro"]==1]
res = stats.mannwhitneyu(no_euro["EU Trust"], euros["EU Trust"])
print(res)
print( " duvera v EU je stejan ve statech s eurem i bez. (nulova hypoteza duvera ve EU ve statech v eurozone je stejna jako ve statech bez eura")