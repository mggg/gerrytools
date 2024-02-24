extern crate csv;
extern crate serde;
extern crate serde_json;

use std::fs::File;
use serde::Deserialize;
use std::error::Error;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct District {
    county_name: String, 
    precinct_name: String,
    election_1: u64,
    election_2: u64,
    assignment: u64,
}


#[derive(Debug, Deserialize)]
struct DistrictingItem(serde_json::Map<String, serde_json::Value>);

#[derive(Debug, Deserialize)]
struct JsonlRecord {
    name: String,
    weight: u32,
    data: serde_json::Value,
    districting: Vec<DistrictingItem>,
}


fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "../main_data.csv";
    let file = File::open(file_path)?;

    let mut district_list = Vec::<District>::new();

    let mut reader = csv::Reader::from_reader(file);
    for result in reader.deserialize() {
        let district: District = result?;
        district_list.push(district);
    }

    let district_map_2: HashMap<(String, String), usize> = district_list.iter().enumerate()
        .map(|(index, dist)| ((dist.county_name.clone(), dist.precinct_name.clone()), index))
        .collect();

    let jsonl_path = "../../examples/output/MN/454190_atlas_gamma0.jsonl";
    let jsonl_file = File::open(jsonl_path)?;
    let jsonl_reader = BufReader::new(jsonl_file);

    let mut wins_counter = vec![0;20];
    let mut step_counter = 0;

    for (index, line_result) in jsonl_reader.lines().enumerate() {
        if index < 3 {
            continue;
        }

        let line = line_result?;
        let record: JsonlRecord = serde_json::from_str(&line)?;
        
        for item in record.districting {
            for (key, value) in &item.0 {
                let assign_districts: Vec<String> = key.trim_start_matches('[')
                    .trim_end_matches(']')
                    .split("\", \"")
                    .map(|s| s.replace("\"", ""))
                    .collect();

                match assign_districts.len() {
                    1 => {
                        let target_county = &assign_districts[0];
                        for dist in district_list.iter_mut() {
                            if &dist.county_name == target_county {
                                dist.assignment = value.as_u64().unwrap_or(0);
                            }
                        }
                    }
                    2 => {
                        if let Some(index) = district_map_2.get(&(assign_districts[0].clone(), assign_districts[1].clone())) {
                            district_list[*index].assignment = value.as_u64().unwrap_or(0);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut counts_election_1 = vec![0;100];
        let mut counts_election_2 = vec![0;100];
        
        for dist in &district_list {
            counts_election_1[dist.assignment as usize] += dist.election_1;
            counts_election_2[dist.assignment as usize] += dist.election_2;
        }

        let num_won: u32 = counts_election_1.into_iter().zip(counts_election_2.into_iter())
                            .collect::<Vec<_>>()
                            .into_iter()
                            .map(|(x,y)| if x > y {1u32} else {0u32})
                            .collect::<Vec<_>>()
                            .into_iter()
                            .sum();
        wins_counter[num_won as usize] += 1;
        step_counter += 1;
        println!("{:?}",wins_counter);
        println!("Number won {}, step {}, running avg {}", num_won, step_counter, wins_counter.iter().enumerate().map(|(a,&b)| a as u32*b as u32).collect::<Vec<u32>>().iter().sum::<u32>() as f64 /step_counter as f64);
    }

    Ok(())
}
