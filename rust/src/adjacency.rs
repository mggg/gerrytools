use crate::{Error, Result};
use std::collections::HashSet;

#[derive(Debug)]
pub(crate) struct CsrAdjacency {
    pub(crate) offsets: Vec<u32>,
    pub(crate) neighbors: Vec<u32>,
    pub(crate) edge_indices: Vec<u32>,
}

pub(crate) fn build_csr_adjacency(node_count: usize, edges: &[(u32, u32)]) -> CsrAdjacency {
    let mut offsets = vec![0_u32; node_count + 1];
    for &(u, v) in edges {
        offsets[u as usize + 1] += 1;
        offsets[v as usize + 1] += 1;
    }
    for node in 0..node_count {
        offsets[node + 1] += offsets[node];
    }

    let mut cursor = offsets.clone();
    let mut neighbors = vec![0_u32; edges.len() * 2];
    let mut edge_indices = vec![0_u32; edges.len() * 2];
    for (edge_index, &(u, v)) in edges.iter().enumerate() {
        let u_slot = cursor[u as usize] as usize;
        neighbors[u_slot] = v;
        edge_indices[u_slot] = edge_index as u32;
        cursor[u as usize] += 1;

        let v_slot = cursor[v as usize] as usize;
        neighbors[v_slot] = u;
        edge_indices[v_slot] = edge_index as u32;
        cursor[v as usize] += 1;
    }

    CsrAdjacency {
        offsets,
        neighbors,
        edge_indices,
    }
}

pub(crate) fn validate_edge_nodes(node_count: usize, edges: &[(u32, u32)]) -> Result<()> {
    let mut seen = HashSet::with_capacity(edges.len());
    if let Some(&(u, v)) = edges
        .iter()
        .find(|&&(u, v)| u as usize >= node_count || v as usize >= node_count)
    {
        return Err(Error::EdgeNodeOutOfRange { u, v, node_count });
    }
    for &(u, v) in edges {
        if u == v {
            return Err(Error::InvalidInput(format!(
                "graph edge ({u}, {v}) is a self-loop"
            )));
        }
        if !seen.insert((u.min(v), u.max(v))) {
            return Err(Error::InvalidInput(format!(
                "graph contains duplicate edge ({u}, {v})"
            )));
        }
    }
    Ok(())
}
