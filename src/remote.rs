use crate::engine::board::*;
use crate::engine::endgame::*;
use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use clap::ArgMatches;
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response};
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpListener;

async fn worker_impl(solve_obj: SolveObj, req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    match req.method() {
        &Method::POST => {}
        method => {
            println!("Unsupported method: {:?}", method);
        }
    }
    let bytes = req.collect().await?.to_bytes();
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

async fn worker_body() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let search_params = SearchParams {
        reduce: false,
        parallel_depth_limit: 16,
        parallel_empties_limit: 16,
        eval_ordering_limit: 15,
        res_cache_limit: 12,
        local_res_cache_limit: 9,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 5,
    };
    let evaluator = Arc::new(Evaluator::load(Path::new("table-220710")).unwrap());
    let res_cache = Arc::new(ResCacheTable::new(256, 65536));
    let eval_cache = Arc::new(EvalCacheTable::new(256, 65536));
    let solve_obj = SolveObj::new(res_cache, eval_cache, evaluator, search_params, 0);

    let addr = SocketAddr::from(([0, 0, 0, 0], 7733));

    let listener = TcpListener::bind(addr).await?;

    println!("Listening on http://{}", addr);

    loop {
        let (tcp, _) = listener.accept().await?;

        let io = TokioIo::new(tcp);

        let solve_obj = solve_obj.clone();

        tokio::task::spawn(async move {
            let service = service_fn(|req| async { worker_impl(solve_obj.clone(), req).await });
            if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                println!("Error serving connection: {:?}", err);
            }
        });
    }
}

pub fn worker(_matches: &ArgMatches) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _ = rt.block_on(worker_body());
}
