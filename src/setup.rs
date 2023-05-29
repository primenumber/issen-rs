use crate::engine::eval::*;
use crate::engine::search::*;
use crate::engine::table::*;
use std::sync::Arc;

pub fn setup_default() -> SolveObj {
    let res_cache = Arc::new(ResCacheTable::new(256, 65536));
    let eval_cache = Arc::new(EvalCacheTable::new(256, 65536));
    let evaluator = Arc::new(Evaluator::new("table-220710"));
    let search_params = SearchParams {
        reduce: false,
        ybwc_depth_limit: 12,
        ybwc_elder_add: 1,
        ybwc_younger_add: 2,
        ybwc_empties_limit: 18,
        eval_ordering_limit: 15,
        res_cache_limit: 11,
        stability_cut_limit: 8,
        ffs_ordering_limit: 6,
        static_ordering_limit: 3,
        use_worker: false,
    };
    SolveObj::new(res_cache, eval_cache, evaluator, search_params)
}
