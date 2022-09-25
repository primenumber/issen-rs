use crate::engine::board::*;
use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use clap::ArgMatches;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

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

async fn worker_impl(
    solve_obj: SolveObj,
    req: Request<Body>,
) -> Result<Response<Body>, Infallible> {
    let bytes = hyper::body::to_bytes(req.into_body()).await.unwrap();
    let query: SolveRequest =
        serde_json::from_str(&String::from_utf8(bytes.to_vec()).unwrap()).unwrap();
    let board = Board::from_base81(&query.board).unwrap();
    let result = solve_inner(
        &mut solve_obj.clone(),
        board,
        query.alpha,
        query.beta,
        false,
    );
    let res_str = serde_json::to_string(&serde_json::json!(SolveResponse {
        result: result.0,
        node_count: result.1.node_count,
        st_cut_count: result.1.st_cut_count,
    }))
    .unwrap();
    Ok(Response::new(res_str.into()))
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}

async fn worker_body() {
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
    let solve_obj = SolveObj::new(res_cache, eval_cache, evaluator, search_params);

    let addr = SocketAddr::from(([0, 0, 0, 0], 7733));

    let make_service = make_service_fn(move |_conn| {
        let context = solve_obj.clone();
        let service = service_fn(move |req| worker_impl(context.clone(), req));
        async move { Ok::<_, Infallible>(service) }
    });

    let server = Server::bind(&addr).serve(make_service);

    let graceful = server.with_graceful_shutdown(shutdown_signal());

    if let Err(e) = graceful.await {
        eprintln!("server error: {}", e);
    }
}

pub fn worker(_matches: &ArgMatches) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(worker_body());
}
