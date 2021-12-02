
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import matplotlib.colors as mcol
import seaborn as sns

my_cmap = mcol.ListedColormap(
    ['#0000ff',
    '#5a00ef',
    '#8000dd',
    '#9c00ca',
    '#b400b4',
    '#ca009c',
    '#dd0080',
    '#ef005a',
    '#ff0000'])

def R_wins_by_district(districts_df, elections):
    # districts_df must be dissolved by districts
    dists_R_wins = [0] * 99
    for index, row in districts_df.iterrows():
        for election in elections:
            if row[f"{election}R"] > row[f"{election}D"]:
                dists_R_wins[index] += 1
    return dists_R_wins

def make_red_blue_plot(plan, elections, state_shape, unit_col, plan_folder=None, plot_folder=None):
    state_gdf = gpd.read_file(state_shape)
    if plan_folder is not None:
        part_df = gpd.read_file(f"{plan_folder}/{plan}.csv")
    else:
        part_df = gpd.read_file(f"{plan}.csv")
    part_df = part_df.rename(columns={'field_2': 'plan_dist'})
    part_df = part_df.rename(columns={'field_1': unit_col})
    part_df[unit_col] = part_df[unit_col].astype(int)
    part_df['plan_dist'] = part_df['plan_dist'].astype(int)
    part_df = part_df.drop(columns=['geometry'])
    part_df = part_df.merge(state_gdf, on=unit_col)
    districts_df = part_df.dissolve(by='plan_dist', aggfunc='sum')
    
    R_wins = R_wins_by_district(districts_df, elections)
    for i, wins in enumerate(R_wins):
        part_df.loc[part_df.plan_dist==i, 'R_dist_wins'] = wins
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    part_df.plot(column='R_dist_wins', ax=ax, cmap=my_cmap)
    districts_df.boundary.plot(ax=ax, color = 'black', linewidth = 0.25)
    ax.axis('off')
    ax.margins(0)
    ax.set_title(plan, fontsize=24)
    ax.apply_aspect()
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=my_cmap), location='bottom', ax=ax, ticks=[0,0.25,0.5,0.75,1])
    cbar.ax.set_xticklabels([0,2,4,6,8])
    if plot_folder is not None:
        plt.savefig(f"{plot_folder}/{plan}_R_wins_map.png")
    else:
        plt.savefig(f"{plan}_R_wins_map.png")
    plt.show()
    return
