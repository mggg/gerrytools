//! Incremental connected-component tracking for the region "parts" statistic.

use super::region::NO_REGION;
use crate::adjacency::CsrAdjacency;
use crate::scoring::delta::{DeltaChange, GenerationStamps};

const NO_COMPONENT: usize = usize::MAX;
pub(crate) const MAX_PACKED_REGIONS: u64 = 1_u64 << 48;

fn component_key(region: usize, district: u16) -> u64 {
    ((region as u64) << 16) | u64::from(district)
}

pub(crate) struct PartColumnState {
    component_of: Vec<usize>,
    position: Vec<usize>,
    components: Vec<Vec<usize>>,
    component_keys: Vec<u64>,
    free_components: Vec<usize>,
    parts: usize,
    visited: GenerationStamps,
    stack: Vec<usize>,
    neighbor_components: Vec<usize>,
    affected_components: Vec<usize>,
    removed_by_component: Vec<Vec<usize>>,
}

impl PartColumnState {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            component_of: vec![NO_COMPONENT; node_count],
            position: vec![0; node_count],
            components: Vec::new(),
            component_keys: Vec::new(),
            free_components: Vec::new(),
            parts: 0,
            visited: GenerationStamps::new(node_count),
            stack: Vec::new(),
            neighbor_components: Vec::new(),
            affected_components: Vec::new(),
            removed_by_component: Vec::new(),
        }
    }

    pub(crate) fn parts(&self) -> usize {
        self.parts
    }

    pub(crate) fn reset(
        &mut self,
        regions: &[usize],
        assignment: &[u16],
        adjacency: &CsrAdjacency,
    ) {
        self.component_of.fill(NO_COMPONENT);
        self.components.clear();
        self.component_keys.clear();
        self.free_components.clear();
        self.affected_components.clear();
        self.removed_by_component.clear();
        self.parts = 0;

        for node in 0..assignment.len() {
            let region = regions[node];
            if region == NO_REGION {
                continue;
            }
            if self.component_of[node] != NO_COMPONENT {
                continue;
            }

            let component =
                self.allocate_component(Vec::new(), component_key(region, assignment[node]));
            self.assign_node(node, component);
            self.stack.push(node);
            while let Some(current) = self.stack.pop() {
                let incident =
                    adjacency.offsets[current] as usize..adjacency.offsets[current + 1] as usize;
                for adjacency_index in incident {
                    let neighbor = adjacency.neighbors[adjacency_index] as usize;
                    if self.component_of[neighbor] == NO_COMPONENT
                        && regions[neighbor] == region
                        && assignment[neighbor] == assignment[node]
                    {
                        self.assign_node(neighbor, component);
                        self.stack.push(neighbor);
                    }
                }
            }
            self.parts += 1;
        }
    }

    pub(crate) fn update(
        &mut self,
        regions: &[usize],
        adjacency: &CsrAdjacency,
        changes: &[DeltaChange],
    ) {
        debug_assert!(self.affected_components.is_empty());
        for change in changes {
            if change.old == change.new || regions[change.node] == NO_REGION {
                continue;
            }
            let component = self.component_of[change.node];
            debug_assert_ne!(component, NO_COMPONENT);
            self.remove_node(change.node, component);
            if self.removed_by_component[component].is_empty() {
                self.affected_components.push(component);
            }
            self.removed_by_component[component].push(change.node);
        }

        while let Some(component) = self.affected_components.pop() {
            let removed = std::mem::take(&mut self.removed_by_component[component]);
            if self.components[component].is_empty() {
                self.release_component(component);
                self.parts -= 1;
            } else {
                self.repair_after_removals(component, &removed, adjacency);
            }
            self.removed_by_component[component] = removed;
            self.removed_by_component[component].clear();
        }

        for change in changes {
            let region = regions[change.node];
            if region == NO_REGION {
                continue;
            }
            if change.old == change.new {
                continue;
            }
            self.add_node(change.node, region, change.new, adjacency);
        }
    }

    fn repair_after_removals(
        &mut self,
        component: usize,
        removed: &[usize],
        adjacency: &CsrAdjacency,
    ) {
        let mut seeds = Vec::new();
        for &node in removed {
            let incident = adjacency.offsets[node] as usize..adjacency.offsets[node + 1] as usize;
            for adjacency_index in incident {
                let neighbor = adjacency.neighbors[adjacency_index] as usize;
                if self.component_of[neighbor] == component {
                    seeds.push(neighbor);
                }
            }
        }
        seeds.sort_unstable();
        seeds.dedup();
        if seeds.len() <= 1 {
            return;
        }

        self.visited.advance();
        let first = self.flood_until_all_seeds(component, seeds[0], &seeds, adjacency);
        if first.1 {
            return;
        }

        let mut fragments = vec![first.0];
        for &seed in &seeds[1..] {
            if !self.visited.is_marked(seed) {
                fragments.push(self.flood_component(component, seed, adjacency));
            }
        }
        let remaining = self.components[component]
            .iter()
            .copied()
            .filter(|&node| !self.visited.is_marked(node))
            .collect::<Vec<_>>();
        for seed in remaining {
            if !self.visited.is_marked(seed) {
                fragments.push(self.flood_component(component, seed, adjacency));
            }
        }

        let retained = fragments
            .iter()
            .enumerate()
            .max_by_key(|(_, fragment)| fragment.len())
            .map(|(index, _)| index)
            .expect("a nonempty component has a surviving fragment");
        let retained_nodes = fragments.swap_remove(retained);
        let key = self.component_keys[component];
        self.components[component] = retained_nodes;
        self.relabel_component(component);
        let added = fragments.len();
        for fragment in fragments {
            self.allocate_component(fragment, key);
        }
        self.parts += added;
    }

    fn flood_until_all_seeds(
        &mut self,
        component: usize,
        start: usize,
        seeds: &[usize],
        adjacency: &CsrAdjacency,
    ) -> (Vec<usize>, bool) {
        let mut nodes = Vec::new();
        let mut reached = 0;
        self.visited.mark(start);
        self.stack.push(start);
        while let Some(node) = self.stack.pop() {
            nodes.push(node);
            if seeds.binary_search(&node).is_ok() {
                reached += 1;
                if reached == seeds.len() {
                    self.stack.clear();
                    return (nodes, true);
                }
            }
            self.push_unvisited_neighbors(node, component, adjacency);
        }
        (nodes, false)
    }

    fn flood_component(
        &mut self,
        component: usize,
        start: usize,
        adjacency: &CsrAdjacency,
    ) -> Vec<usize> {
        let mut nodes = Vec::new();
        self.visited.mark(start);
        self.stack.push(start);
        while let Some(node) = self.stack.pop() {
            nodes.push(node);
            self.push_unvisited_neighbors(node, component, adjacency);
        }
        nodes
    }

    fn push_unvisited_neighbors(
        &mut self,
        node: usize,
        component: usize,
        adjacency: &CsrAdjacency,
    ) {
        let incident = adjacency.offsets[node] as usize..adjacency.offsets[node + 1] as usize;
        for adjacency_index in incident {
            let neighbor = adjacency.neighbors[adjacency_index] as usize;
            if self.component_of[neighbor] == component && !self.visited.is_marked(neighbor) {
                self.visited.mark(neighbor);
                self.stack.push(neighbor);
            }
        }
    }

    fn add_node(&mut self, node: usize, region: usize, district: u16, adjacency: &CsrAdjacency) {
        self.neighbor_components.clear();
        let mut first_component = NO_COMPONENT;
        let key = component_key(region, district);
        let incident = adjacency.offsets[node] as usize..adjacency.offsets[node + 1] as usize;
        for adjacency_index in incident {
            let neighbor = adjacency.neighbors[adjacency_index] as usize;
            let component = self.component_of[neighbor];
            if component == NO_COMPONENT || self.component_keys[component] != key {
                continue;
            }
            if first_component == NO_COMPONENT {
                first_component = component;
            } else if component != first_component && !self.neighbor_components.contains(&component)
            {
                self.neighbor_components.push(component);
            }
        }
        self.attach_node(node, key, first_component);
    }

    fn attach_node(&mut self, node: usize, key: u64, first_component: usize) {
        if first_component == NO_COMPONENT {
            let component = self.allocate_component(Vec::new(), key);
            self.assign_node(node, component);
            self.parts += 1;
            return;
        }
        if self.neighbor_components.is_empty() {
            self.assign_node(node, first_component);
            return;
        }

        let mut target = first_component;
        for &component in &self.neighbor_components {
            if self.components[component].len() > self.components[target].len() {
                target = component;
            }
        }
        self.assign_node(node, target);
        if first_component != target {
            let members = std::mem::take(&mut self.components[first_component]);
            self.free_components.push(first_component);
            for member in members {
                self.assign_node(member, target);
            }
        }
        for index in 0..self.neighbor_components.len() {
            let source = self.neighbor_components[index];
            if source == target {
                continue;
            }
            let members = std::mem::take(&mut self.components[source]);
            self.free_components.push(source);
            for member in members {
                self.assign_node(member, target);
            }
        }
        self.parts -= self.neighbor_components.len();
    }

    fn allocate_component(&mut self, members: Vec<usize>, key: u64) -> usize {
        let component = match self.free_components.pop() {
            Some(component) => {
                debug_assert!(self.components[component].is_empty());
                component
            }
            None => {
                self.components.push(Vec::new());
                self.component_keys.push(key);
                self.removed_by_component.push(Vec::new());
                self.components.len() - 1
            }
        };
        self.components[component] = members;
        self.component_keys[component] = key;
        self.relabel_component(component);
        component
    }

    fn release_component(&mut self, component: usize) {
        debug_assert!(self.components[component].is_empty());
        self.free_components.push(component);
    }

    fn assign_node(&mut self, node: usize, component: usize) {
        self.component_of[node] = component;
        self.position[node] = self.components[component].len();
        self.components[component].push(node);
    }

    fn remove_node(&mut self, node: usize, component: usize) {
        let position = self.position[node];
        self.components[component].swap_remove(position);
        if let Some(&moved) = self.components[component].get(position) {
            self.position[moved] = position;
        }
        self.component_of[node] = NO_COMPONENT;
    }

    fn relabel_component(&mut self, component: usize) {
        for (position, &node) in self.components[component].iter().enumerate() {
            self.component_of[node] = component;
            self.position[node] = position;
        }
    }
}
