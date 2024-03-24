import pandas as pd
import numpy as np 
from skbio.stats.ordination import pcoa  
from scipy.spatial.distance import braycurtis, pdist, squareform
import matplotlib.pyplot as plt
df_species = pd.read_csv("species_master_without_9.tsv", sep = "\t", index_col=0)


species = df_species.columns.tolist()
species = species[0:202]
days = df_species["Days"]
max_day = days.to_numpy(dtype=int).max()
print(max_day)
#print(species)
#print(df_species.columns)
df_species_subset = df_species[species]
#print(df_species_subset)


#grouped = df_species_subset.groupby("Player")
#sliced_df = pd.concat([slice_data_for_PCoA(group_df) for _, group_df in grouped], ignore_index = True)



#sliced_df=sliced_df.loc[:,species]
#print(pdist(sliced_df))
matrix = squareform(pdist(df_species_subset, metric = "braycurtis"))


pcoa_results = pcoa(matrix)
pcoa_df = pcoa_results.samples[['PC1', 'PC2']]
prop_exp = pcoa_results.proportion_explained[["PC1", "PC2"]]
print(prop_exp)
#print(df_species["Player"])
pcoa_df["Player"] = df_species["Player"].to_numpy(dtype = "str")
pcoa_df["Days"] = df_species["Days"].to_numpy(dtype="int")
print(pcoa_df)
plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "1", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "1", "PC2"],c = 'blue')

plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "4", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "4", "PC2"],c = 'orange')
plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "5", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "5", "PC2"],c = 'green')

plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "8", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "8", "PC2"],c = 'red')

plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "9", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "9", "PC2"],c = 'purple')


plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "16", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "16", "PC2"],c = 'black')

def encircle2(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    mean = np.mean(p, axis=0)
    d = p-mean
    r = np.max(np.sqrt(d[:,0]**2+d[:,1]**2 ))
    circ = plt.Circle(mean, radius=1.05*r,**kw)
    ax.add_patch(circ)

encircle2(pcoa_df.loc[pcoa_df["Player"] == "1", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "1", "PC2"], ec="blue", fc="none")

encircle2(pcoa_df.loc[pcoa_df["Player"] == "4", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "4", "PC2"], ec="orange", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "5", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "5", "PC2"], ec="green", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "8", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "8", "PC2"], ec="red", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "9", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "9", "PC2"], ec="purple", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "16", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "16", "PC2"], ec="black", fc="none")

plt.gca().relim()
plt.gca().autoscale_view()
plt.xlabel(f"PCo1 {prop_exp[0]*100:.2f}%")
plt.ylabel(f"PCo2 {prop_exp[1]*100:.2f}%")
plt.legend(["1","4","5","8","9","16"],title="Player")
plt.savefig("PcoA_all_players.png", dpi = 300, bbox_inches="tight")

plt.clf()
plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "1", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "1", "PC2"],c = 'blue',
    alpha=pcoa_df.loc[pcoa_df["Player"] == "1", "Days"]/pcoa_df.loc[pcoa_df["Player"] == "1", "Days"].max())

plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "4", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "4", "PC2"],c = 'orange',
    alpha=pcoa_df.loc[pcoa_df["Player"] == "4", "Days"]/pcoa_df.loc[pcoa_df["Player"] == "4", "Days"].max())

plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "5", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "5", "PC2"],c = 'green',
    alpha=pcoa_df.loc[pcoa_df["Player"] == "5", "Days"]/pcoa_df.loc[pcoa_df["Player"] == "5", "Days"].max())


plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "8", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "8", "PC2"],c = 'red',
    alpha=pcoa_df.loc[pcoa_df["Player"] == "8", "Days"]/pcoa_df.loc[pcoa_df["Player"] == "8", "Days"].max())


plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "9", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "9", "PC2"],c = 'purple',
    alpha=pcoa_df.loc[pcoa_df["Player"] == "9", "Days"]/pcoa_df.loc[pcoa_df["Player"] == "9", "Days"].max())



plt.scatter(pcoa_df.loc[pcoa_df["Player"] == "16", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "16", "PC2"],c = 'black',
    alpha=pcoa_df.loc[pcoa_df["Player"] == "16", "Days"]/pcoa_df.loc[pcoa_df["Player"] == "16", "Days"].max())



encircle2(pcoa_df.loc[pcoa_df["Player"] == "1", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "1", "PC2"], ec="blue", fc="none")

encircle2(pcoa_df.loc[pcoa_df["Player"] == "4", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "4", "PC2"], ec="orange", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "5", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "5", "PC2"], ec="green", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "8", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "8", "PC2"], ec="red", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "9", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "9", "PC2"], ec="purple", fc="none")
encircle2(pcoa_df.loc[pcoa_df["Player"] == "16", "PC1"], 
    pcoa_df.loc[pcoa_df["Player"] == "16", "PC2"], ec="black", fc="none")

plt.gca().relim()
plt.gca().autoscale_view()
plt.xlabel(f"PCo1 {prop_exp[0]*100:.2f}%")
plt.ylabel(f"PCo2 {prop_exp[1]*100:.2f}%")
#plt.legend(["1","4","5","8","9","16"],title="Player")
plt.savefig("PcoA_all_players_w_days.png", dpi = 300, bbox_inches="tight")



