
import sympy as sp
import itertools
from compiler import construct_symbolic_matrix
import fast_decomposer as fastdec
import deep_decomposer as deepdec


def matrix_entry_type(M):
    """
    Detect whether matrix has complex entries.
    """
    for x in M:
        if sp.I in sp.expand(x).free_symbols or x.has(sp.I):
            return "complex"
    return "integer"



def get_all_paths_generalized(tree):
    """
    Input:
        generalized symbolic tree or numeric tree

    Output:
        all solve/combine paths that will be evaluated later to be solvedl
    """

    paths = []

    def recurse(node):
        """
        Returns all possible path suffixes from this node.
        """

        matrix_info = {
            "matrix": node.get("matrix", None)
        }

        if "matrix_examples" in node:
            matrix_info["matrix_examples"] = node["matrix_examples"]

        # -----------------------------------
        # Leaf: solve directly
        # -----------------------------------
        if not node["children"]:
            return [[(
                node["dim"],
                node["variance"],
                matrix_entry_type(
                    node.get("matrix", node.get("matrix_examples", [sp.Matrix()])[0])
                ),
                "solve",
                matrix_info
            )]]

        # -----------------------------------
        # Option 1: Solve current node directly
        # -----------------------------------
        all_paths = [[(
            node["dim"],
            node["variance"],
            matrix_entry_type(
                node.get("matrix", node.get("matrix_examples", [sp.Matrix()])[0])
            ),
            "solve",
            matrix_info
        )]]

        # -----------------------------------
        # Option 2: Combine children recursively
        # -----------------------------------
        child_paths_lists = [
            recurse(child)
            for child in node["children"]
        ]

        for combination in itertools.product(*child_paths_lists):

            combined_path = [(
                node["dim"],
                node["variance"],
                matrix_entry_type(
                    node.get("matrix", node.get("matrix_examples", [sp.Matrix()])[0])
                ),
                "combine",
                {
                    "transform": node.get("transform", None),
                    "transform_examples": node.get("transform_examples", None)
                }
            )]

            for child_path in combination:
                combined_path.extend(child_path)

            all_paths.append(combined_path)

        return all_paths

    return recurse(tree)




# ============================================================
# Evaluate symbolic expression
# ============================================================

def eval_symbolic(expr, n_value):
    if isinstance(expr, (int, float)):
        return float(expr)

    n = sp.Symbol('n')
    return float(sp.sympify(expr).subs(n, n_value))


# ============================================================
# Evaluate one path
# ============================================================

def evaluate_path_beta(path, n_value, q, beta_func, verbose=True):

    betas = []
    has_complex = False
    critical_node = None

    for idx, step in enumerate(path):

        dim, variance, typ, role, meta = step

        if role != "solve":
            continue

        dim_val = int(eval_symbolic(dim, n_value))
        var_val = eval_symbolic(variance, n_value)

        # -----------------------------------
        # Complex doubles dimension
        # -----------------------------------
        if typ == "complex":
            dim_val *= 2
            has_complex = True
        if(verbose):
            print("testing for: {}, {}, {} ".format(dim_val, q, var_val))
        result = beta_func(q,dim_val, var_val,verbose=True)
        beta = result[0]
#         result = beta_func(q,dim_val, var_val)
#         beta = result[0]

        betas.append((beta, idx))

    if not betas:
        return None

    max_beta, critical_idx = max(betas, key=lambda x: x[0])

    return {
        "path": path,
        "beta": max_beta,
        "complex": has_complex,
        "critical_node": critical_idx
    }


# ============================================================
# Best path for one tree
# ============================================================

def best_beta_for_tree(paths, n_value, q, beta_func,verbose=True):

    evaluated = []

    for path in paths:
        res = evaluate_path_beta(path, n_value, q, beta_func,verbose=verbose)

        if res is not None:
            evaluated.append(res)

    if not evaluated:
        return None

    # -----------------------------------
    # Tie-break: prefer complex on equal beta
    # -----------------------------------
    best_overall = min(
        evaluated,
        key=lambda x: (x["beta"], -int(x["complex"]))
    )

    integer_only = [
        x for x in evaluated
        if not x["complex"]
    ]

    best_integer = None
    if integer_only:
        best_integer = min(
            integer_only,
            key=lambda x: x["beta"]
        )

    return {
        "best_overall": best_overall,
        "best_integer": best_integer
    }


# ============================================================
# Global optimizer
# ============================================================

def best_global_beta(generalized_trees, beta_func, path_builder,verbose=True,n_value=None, q=None):

    global_best = None
    global_best_integer = None

    for tree_idx, tree in enumerate(generalized_trees):

        paths = path_builder(tree)

        result = best_beta_for_tree(
            paths=paths,
            n_value=n_value,
            q=q,
            beta_func=beta_func,
            verbose=verbose
        )

        if result is None:
            continue

        # -----------------------------------
        # Global best overall
        # Tie-break prefers complex
        # -----------------------------------
        cand = result["best_overall"]

        if (
            global_best is None
            or (cand["beta"] < global_best["beta"])
            or (
                cand["beta"] == global_best["beta"]
                and cand["complex"]
                and not global_best["complex"]
            )
        ):
            global_best = {
                **cand,
                "tree_index": tree_idx
            }

        # -----------------------------------
        # Global best integer only
        # -----------------------------------
        if result["best_integer"] is not None:

            cand_int = result["best_integer"]

            if (
                global_best_integer is None
                or cand_int["beta"] < global_best_integer["beta"]
            ):
                global_best_integer = {
                    **cand_int,
                    "tree_index": tree_idx
                }

    return {
        "best_global": global_best,
        "best_global_integer": global_best_integer,
        "concrete_beta": global_best_integer['beta']
    }



    

def estimate_security(n, q, base_var, mul_function, tage="", level=1, lattice="NTRU" verbose=True, mul_type="LL", filename="examples.py", variable='h'):
    """
    Input:  - The parameters to be used to evaluate security.
    Output: - Beta which is going to be used for security estimation
    """
    
     H = compiler.construct_symbolic_matrix(
        filename  = filename,
        funcname  = mul_function,
        dimension = n,
        mul_type  = mul_type,
        variable  = 'h'
        )
    match level:
        case 1:
            decomposer = fastdec.SymbolicDecomposer(
            symbolic_matrix=H,
            n=n,
            q=q,
            base_var=base_var
        )
        case 2:
            decomposer = deepdec.PrimaryDecomposer(
            symbolic_matrix=H,
            n=n,
            q=q,
            base_var=base_var
        )
        case 3:
            decomposer = deepdec.ExhuastivDecomposer(
            symbolic_matrix=H,
            n=n,
            q=q,
            base_var=base_var
        )
        case _:
            return "Invalid level"

        full_trees = decomposer.get_full_trees(verbose=verbose)
        result = best_global_beta(full_trees, beta_func, path_builder,verbose=verbose,n_value=n, q=q)
        
        return result




################################# Takes as an input toy example generalize, then get the security #####################
def get_the_generalized_tree(n_list, q_list, base_var, mul_function, tage="", level=1, verbose=True, mul_type="LL", filename="examples.py", variable='h'):
    """
    Input:    provide list if you want to run for toy examples and generalize lateron otherwise give single n   and q values.
             - n_list: the list on n values
             - q_list: the list of q values
              base_var: the base var of the private key distribution
             - mul_function: the multiplication function name
             - tag: a string to describe the multiplication function
             - level: indicating the level of decomposing (level=1) is the simplest
    Output: - all the pathes for this list 
    """

    trees = {}
    for idx in range(len(n_list)):
        n = n_list[idx]
        q = q_list[idx]
        
        H = compiler.construct_symbolic_matrix(
        filename  = filename,
        funcname  = mul_function,
        dimension = n,
        mul_type  = mul_type,
        variable  = 'h'
        )
        
        match level:
        case 1:
            decomposer = fastdec.SymbolicDecomposer(
            symbolic_matrix=H,
            n=n,
            q=q,
            base_var=base_var
        )
        case 2:
            decomposer = deepdec.PrimaryDecomposer(
            symbolic_matrix=H,
            n=n,
            q=q,
            base_var=base_var
        )
        case 3:
            decomposer = deepdec.ExhuastivDecomposer(
            symbolic_matrix=H,
            n=n,
            q=q,
            base_var=base_var
        )
        case _:
            return "Invalid level"
      
        full_trees = decomposer.get_full_trees(verbose=verbose)


        trees[n] = []

        for i, tree in enumerate(full_trees):

            trees[n].append({
                "tree_id": i,
                "tree": tree
            })
            
    return trees
    
