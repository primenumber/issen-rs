use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use std::path::Path;
use std::sync::Arc;

pub fn setup_default() -> SolveObj {
    let res_cache = Arc::new(ResCacheTable::new(2048, 16384));
    let eval_cache = Arc::new(EvalCacheTable::new(2048, 16384));
    let evaluator = Arc::new(Evaluator::load(Path::new("table-220710")).unwrap());
    let search_params = SearchParams {
        reduce: false,
        parallel_depth_limit: 16,
        parallel_empties_limit: 16,
        eval_ordering_limit: 15,
        res_cache_limit: 14,
        local_res_cache_limit: 10,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 5,
    };
    SolveObj::new(res_cache, eval_cache, evaluator, search_params, 0)
}
