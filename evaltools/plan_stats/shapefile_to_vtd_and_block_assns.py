import geopandas as gpd 
import pandas as pd
import math
import tqdm
import os
import maup
from gerrychain import Partition, Graph, GeographicPartition
from gerrychain.updaters import cut_edges, Tally, compactness
import gerrychain
import warnings; warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)

#vtd splits
def vtd_splits(vtd_assignment_csv, block_assignment_csv, plan_shapefile, vtd_shapefile, block_shapefile):
    
    
  vtd_assignments = pd.read_csv(vtd_assignment_csv)
  vtds = gpd.read_file(vtd_shapefile)
  
  district_vtds= gpd.read_file(plan_shapefile).to_crs(vtds.crs)
  district_vtds["DISTRICT"] = district_vtds["DISTRICT"].astype('int')

  blocks = gpd.read_file(block_shapefile).to_crs(vtds.crs)
  block_assignments = pd.read_csv(block_assignment_csv)
  block_vtd_mapping = maup.assign(blocks, vtds)
  block_district_mapping = maup.assign(blocks, district_vtds)
  
  if int(min(block_district_mapping)) != int(min(district_vtds.DISTRICT)): 
    mapping_diff = int(min(district_vtds.DISTRICT)) - int(min(block_district_mapping))
    block_district_mapping += 1

  blocks["VTD"] = block_vtd_mapping
  blocks["DISTRICT"] = block_district_mapping.astype('int')

  
  if "TOTPOP" in vtds.columns: 
        vtd_pop_key = "TOTPOP"
  elif "TOTPOP20" in vtds.columns:
        vtd_pop_key = "TOTPOP20"

  if "TOTPOP" in blocks.columns:
    block_pop_key = "TOTPOP"
  elif "TOTPOP20" in blocks.columns:
    block_pop_key = "TOTPOP20"
    
  vtd_split_count = 0
  displaced_area = []
  displaced_pop = [] 
  total_area_impacted_by_splits = 0
  reassigned_area = 0
  total_pop_impacted_by_splits = 0
  reassigned_pop = 0


  for idx, vtd in tqdm.tqdm(vtds.iterrows()):
    district = int(vtd_assignments["assignment"][idx])
    district_row = district_vtds[district_vtds["DISTRICT"] == district].iloc[0]
    leftover = vtd.geometry - district_row.geometry
        
    if round(1-leftover.area/vtd.geometry.area, 4) < 0.99: 
        secondary_district = 0
        secondary_pop_split = 0
            
        pop_split = sum(blocks[block_pop_key][(blocks["DISTRICT"] == district) & (blocks["VTD"] == idx)])
            
        for ix, dst in district_vtds.iterrows(): 
            if(dst["geometry"].overlaps(vtd.geometry) and dst["DISTRICT"] != district):
                secondary_district = int(dst["DISTRICT"])
                secondary_pop_split = sum(blocks[block_pop_key][(blocks["DISTRICT"] == secondary_district) & (blocks["VTD"] == idx)])
            
            pop_in_district = sum(blocks[block_pop_key][blocks["VTD"] == idx])
            
        if(min(pop_split, secondary_pop_split) >= 10):
            vtd_split_count += 1
            reassigned_area += round(leftover.area, 4))
            reassigned_pop += secondary_pop_split
            displaced_area.append("VTD: " + str(vtd["GEOID20"]) + " area splits. District " + str(district) + " contains " + str(round(1-leftover.area/vtd.geometry.area, 4)) + " proportion of area, while District " + str(secondary_district) + " contains " + str(round(leftover.area/vtd.geometry.area, 4)) + ". Total area of vtd = " + str(vtd.geometry.area)) 
            displaced_pop.append("VTD: " + str(vtd["GEOID20"]) + " population splits. District " + str(district) + " contains " + str(pop_split) + " people , while District " + str(secondary_district) + " contains " + str(secondary_pop_split) + ". VTD has " + str(vtd[vtd_pop_key]) + " total people.")

  crs_ref = str(vtds.crs) + " more info at: https://epsg.io/" + str(vtds.crs).split(":")[1]
  return [vtd_split_count, displaced_area, displaced_pop, reassigned_area, reassigned_pop, crs_ref]


def plan_stat_report(assignment_csv, block_shapefile, block_graph): 
  '''
  assignment_csv: Can contain the plan assignment on VTDs or blocks. Column 1 is GEOID, column 2 is District number. 
  block_shapefile: Shapefile for the state on blocks. 
  block_graph: Graph from json file of block dual graph. 
  '''
  df = pd.read_csv(assignment_csv)
  shp = gpd.read_file(block_shapefile)
  df["GEOID20"] = df["GEOID20"].astype(str)
  shp = shp.merge(df, on = "GEOID20")
  assign = dict(shp["assignment"])

  state_partition = Partition(
    block_graph,
    assignment= assign,
    updaters={
        "cut_edges": cut_edges, 
        "population": Tally("TOTPOP20")

    }
  )

  state_geo_partition = GeographicPartition(
      block_graph, 
      assignment = assign, 
      updaters= {
          "compactness": compactness
      }
  )

  ideal_pop = round(sum(state_partition.population.values())/len(state_partition.population))
  ideal_pop = round(ideal_pop)
    
  pos_pop_dev = 0
  max_pos_dev_district = 0
    
  neg_pop_dev = 0
  max_neg_dev_district = 0
  for district, pop in state_partition.population.items():
    if (ideal_pop-pop)/ideal_pop > pos_pop_dev:
        pos_pop_dev = (ideal_pop-pop)/ideal_pop
        max_pos_dev_district = district
    if (ideal_pop-pop)/ideal_pop < neg_pop_dev:
        neg_pop_dev = (ideal_pop-pop)/ideal_pop
        max_neg_dev_district = district

  polsby = sum(gerrychain.metrics.polsby_popper(state_geo_partition).values()) / len(state_geo_partition.parts)

  proportion_of_cut_edges = len(state_partition.cut_edges) / len(state_partition.graph.edges)
  return {"Proportion of cut_edges": proportion_of_cut_edges, "Cut Edges": len(state_partition.cut_edges), "Total Edges": len(state_partition.graph.edges), "Polsby popper": polsby, "Max Positive Populaton Deviation": pos_pop_dev, "District of Max Positive Population Deviation": max_pos_dev_district, "Max Negative Population Deviation": neg_pop_dev, "District of Max Negative Population Deviation": max_neg_dev_district}

def plan_shapefile_to_assignment(plan_shapefile, vtd_shapefile, block_shapefile):
  '''
  plan_shapefile: Shapefile for the plan, usually pulled from redistricting site. 
  vtd_shapefile: Shapefile for the state on vtds. 
  block_shapefile: Shapefile for the state on blocks. 
  '''
  state_vtds = gpd.read_file(vtd_shapefile)
  state_blocks = gpd.read_file(block_shapefile).to_crs(state_vtds.crs)
    
  shp = gpd.read_file(plan_shapefile).to_crs(state_vtds.crs)
  bck_assignment = maup.assign(state_blocks, shp)
  vtd_assignment = maup.assign(state_vtds, shp)
    
  assignment_filename_prefix = plan_shapefile.split("/")[-1].split(".")[0].replace(" ", "_")
    
  state_blocks["assignment"] = bck_assignment
  state_vtds["assignment"] = vtd_assignment
    
  state_blocks["assignment"] = state_blocks["assignment"].apply(lambda x: shp.iloc[int(x)]["DISTRICT"] if not math.isnan(x) else None)
  state_vtds["assignment"] = state_vtds["assignment"].apply(lambda x: shp.iloc[int(x)]["DISTRICT"] if not math.isnan(x) else None)
    
  state_blocks[[x for x in state_blocks.columns if "POP" in x or "VAP" in x]].groupby(state_blocks["assignment"]).sum().to_csv(assignment_filename_prefix + "_bck_stats.csv")
  state_vtds[[x for x in state_vtds.columns if "POP" in x or "VAP" in x]].groupby(state_vtds["assignment"]).sum().to_csv(assignment_filename_prefix + "_vtd_stats.csv")
    
  state_blocks = state_blocks["assignment"].astype(int)
  state_vtds = state_vtds["assignment"].astype(int)

  state_blocks = state_blocks["GEOID20"].astype(str)
  state_vtds = state_vtds["GEOID20"].astype(str)

  state_blocks[["GEOID20", "assignment"]].dropna().to_csv(assignment_filename_prefix + "_bck_dropped.csv", index=False)
  state_vtds[["GEOID20", "assignment"]].dropna().to_csv(assignment_filename_prefix + "_vtd_dropped.csv", index=False)
 
  state_blocks[["GEOID20", "assignment"]].to_csv(assignment_filename_prefix + "_bck.csv", index=False)
  state_vtds[["GEOID20", "assignment"]].to_csv(assignment_filename_prefix + "_vtd.csv", index=False)

  state_blocks.to_file(assignment_filename_prefix + "_on_blocks.shp")
  state_vtds.to_file(assignment_filename_prefix + "_on_vtds.shp")

def generate_plan_report_from_csv(csv_prefix, block_json, block_shapefile, vtd_shapefile):
  #version to be used when assignment csvs have already been generated  
  '''
  csv_prefix: Prefix for the plan files, i.e. "HD-A5"
  block_json: Json file for block dual graph of state
  block_shapefile: Shapefile for the state on blocks
  vtd_shapefile: Shapefile for state on vtds
  '''
  with tqdm.tqdm(total=100) as pbar:
      block_graph = Graph.from_json(block_json)
      pbar.update(100/3)
      block_assignment_csv =  csv_prefix + "_bck.csv"
      vtd_assignment_csv = csv_prefix + "_vtd.csv"
      plan_shapefile = csv_prefix + ".shp"

      stat_dict = {}
      stat_dict = plan_stat_report(block_assignment_csv, block_shapefile, block_graph)
      pbar.update(100/3)
      vtd_split_stats = vtd_splits(vtd_assignment_csv, block_assignment_csv, plan_shapefile, vtd_shapefile, block_shapefile)
      pbar.update(100/3)
      filename = csv_prefix + "_stat_report.txt"
      plan_report = open(filename, "w")
      plan_report.write(csv_prefix + " stats on blocks (vtd crs is " + vtd_split_stats[5] + "):\n")
      for stat_name, stat in stat_dict.items():
        stats = "{}: {}".format(stat_name, stat)
        plan_report.write(stats + "\n")
      plan_report.write("Number of vtd splits: " + str(vtd_split_stats[0]) + "\n")
      plan_report.write("Reassigned area: " + str(vtd_split_stats[3]) + "\n")
      plan_report.write("Reassigned population: " + str(vtd_split_stats[4]) + "\n")
      for i in range(len(vtd_split_stats[1])):
        plan_report.write(vtd_split_stats[1][i] + "\n")
        plan_report.write(vtd_split_stats[2][i] + "\n")
      plan_report.close()

def generate_plan_report_from_shp(plan_shapefile, block_json, block_shapefile, vtd_shapefile):
      '''
      plan_shapefile: Shapefile for plan
      block_json: Json file for block dual graph of state
      block_shapefile: Shapefile for the state on blocks
      vtd_shapefile: Shapefile for state on vtds
      '''
    plan_shapefile_to_assignment(plan_shapefile, vtd_shapefile, block_shapefile)
    

    csv_prefix = plan_shapefile.split(".shp")[0]
    generate_plan_report_from_csv(csv_prefix, block_json, block_shapefile, vtd_shapefile)
  
