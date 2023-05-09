import pandas
import numpy
import matplotlib.pyplot as plt

def swing(row):
    
    
    if row["change"]== 0:
        return "no change"
    elif row["change"] ==1 and row["party_simplified"] == "REPUBLICAN":
        return "swing to republican"
    else:
        return "swing to demokrat"

df = pandas.read_csv("1976-2020-president.csv")
df =df[["year", "state", "party_simplified", "candidatevotes","candidate","totalvotes"]]
df["Rank"] = df.groupby(["state","year"])["candidatevotes"].rank(method = "min", ascending=False)
df_winners = df[df["Rank"] == 1]
df_winners = df_winners.sort_values(["state", "year"])
df_winners["previous_winner_party"] = df_winners.groupby("state")["party_simplified"].shift(1)

df_winners["change"] = numpy.where(df_winners["party_simplified"] == df_winners["previous_winner_party"], 0, 1)
df_winners_filtered = df_winners[df_winners['year'] > 1976]
data_pivot = df_winners_filtered.groupby(["state"])["change"].sum()
data_pivot = pandas.DataFrame(data_pivot)
data_pivot = data_pivot.sort_values("change", ascending=False)

data_pivot= data_pivot.iloc[:10]
data_pivot.plot(kind = "bar")
plt.ylabel("Numberr of changes")
plt.xlabel("State")
# plt.show()

df_winner_second = df[df["Rank"] <=2]
df_winner_second= df_winner_second.sort_values(["state", "year","Rank"])
df_winner_second["votes second candidate"] = df_winner_second["candidatevotes"].shift(-1)
df_winner_second = df_winner_second[df_winner_second["Rank"] ==1]
df_winner_second["margin"] =df_winner_second["candidatevotes"]-df_winner_second["votes second candidate"]
df_winner_second["relative margin"] = df_winner_second["margin"]/ df_winner_second["totalvotes"]
df_winner_second = df_winner_second.sort_values(["relative margin"])

df_winners_filtered["swing"] =df_winners_filtered.apply(swing, axis =1)
df_winners_filtered= df_winners_filtered.sort_values("year")


df_pivot_swing= df_winners_filtered[["year","state","swing"]]
df_pivot_swing= df_pivot_swing.groupby(["year"])["swing"].value_counts()
print(df_pivot_swing)