use crate::board::*;
use crate::eval::*;
use crate::search::*;
use crate::table::*;
use clap::ArgMatches;
use futures::executor::ThreadPool;
use std::sync::Arc;
use tide::prelude::*;
use tide::{Body, Request};

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

async fn worker_impl() -> tide::Result<()> {
    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 12,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 16,
        eval_ordering_limit: 15,
        res_cache_limit: 11,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    let evaluator = Arc::new(Evaluator::new("table-220710"));
    let res_cache = ResCacheTable::new(256, 65536);
    let eval_cache = EvalCacheTable::new(256, 65536);
    let pool = ThreadPool::new().unwrap();
    let solve_obj = SolveObj::new(
        res_cache.clone(),
        eval_cache.clone(),
        evaluator.clone(),
        search_params.clone(),
        pool.clone(),
    );
    //tide::log::start();
    let mut app = tide::with_state(solve_obj);
    app.with(tide::log::LogMiddleware::new());
    app.at("/").post(|mut req: Request<SolveObj>| async move {
        let mut solve_obj = req.state().clone();
        let query: SolveRequest = req.body_json().await?;
        let board = Board::from_base81(&query.board).unwrap();
        let result = solve_inner(&mut solve_obj, board, query.alpha, query.beta, false);
        Body::from_json(&SolveResponse {
            result: result.0,
            node_count: result.1.node_count,
            st_cut_count: result.1.st_cut_count,
        })
    });
    app.listen("0.0.0.0:7733").await?;
    Ok(())
}

pub fn worker(_matches: &ArgMatches) {
    async_std::task::block_on(worker_impl()).unwrap();
}
