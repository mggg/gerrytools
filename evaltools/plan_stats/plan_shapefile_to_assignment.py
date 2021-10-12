# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 19:11:36 2021

@author: crich
"""
import geopandas as gpd
import maup
import pandas as pd
import math

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