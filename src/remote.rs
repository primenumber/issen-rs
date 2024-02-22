use crate::engine::board::*;
use crate::engine::endgame::*;
use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use clap::ArgMatches;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

async fn worker_impl(solve_obj: SolveObj, req: Request<Body>) -> Result<Response<Body>, Infallible> {
    let bytes = hyper::body::to_bytes(req.into_body()).await.unwrap();
    let query: SolveRequest = serde_json::from_str(&String::from_utf8(bytes.to_vec()).unwrap()).unwrap();
    let board = Board::from_base81(&query.board).unwrap();
    let result = solve_inner(
        &mut solve_obj.clone(),
        board,
        (query.alpha, query.beta),
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
        parallel_depth_limit: 16,
        parallel_empties_limit: 16,
        eval_ordering_limit: 15,
        res_cache_limit: 12,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 5,
    };
    let evaluator = Arc::new(Evaluator::load(Path::new("table-220710")).unwrap());
    let res_cache = Arc::new(ResCacheTable::new(256, 65536));
    let eval_cache = Arc::new(EvalCacheTable::new(256, 65536));
    let solve_obj = SolveObj::new(res_cache, eval_cache, evaluator, search_params, 0);

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
