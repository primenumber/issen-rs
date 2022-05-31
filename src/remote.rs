use tide::prelude::*;
#[derive(Debug, Serialize, Deserialize)]
pub struct SolveRequest {
    pub board: String,
    pub alpha: i8,
    pub beta: i8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveResponse {
    pub result: i8,
    pub node_count: usize,
    pub st_cut_count: usize,
}
