import itertools
import sympy as sp

import compiler
from compiler import AlgebraCompiler

import fast_decomposer as fastdec
#import deep_decomposer as deepdec

from LWE_estimator import estimator
from math import sqrt




class MatrixUtilities:

    @staticmethod
    def matrix_entry_type(M):
        """
        Detect whether matrix has complex entries.
        """

        for x in M:

            if (
                sp.I in sp.expand(x).free_symbols
                or x.has(sp.I)
            ):
                return "complex"

        return "integer"




class PathExtractor:

    @staticmethod
    def get_all_paths_generalized(tree):
        """
        Extract all solve/combine paths from a decomposition tree.
        """

        def recurse(node):

            matrix_info = {
                "matrix": node.get("matrix", None)
            }

            if "matrix_examples" in node:

                matrix_info["matrix_examples"] = (
                    node["matrix_examples"]
                )

    

            if not node["children"]:

                return [[(
                    node["dim"],
                    node["variance"],
                    MatrixUtilities.matrix_entry_type(
                        node.get(
                            "matrix",
                            node.get(
                                "matrix_examples",
                                [sp.Matrix()]
                            )[0]
                        )
                    ),
                    "solve",
                    matrix_info
                )]]

          

            all_paths = [[(
                node["dim"],
                node["variance"],
                MatrixUtilities.matrix_entry_type(
                    node.get(
                        "matrix",
                        node.get(
                            "matrix_examples",
                            [sp.Matrix()]
                        )[0]
                    )
                ),
                "solve",
                matrix_info
            )]]

        

            child_paths_lists = [

                recurse(child)

                for child in node["children"]
            ]

            for combination in itertools.product(
                *child_paths_lists
            ):

                combined_path = [[
                    node["dim"],
                    node["variance"],
                    MatrixUtilities.matrix_entry_type(
                        node.get(
                            "matrix",
                            node.get(
                                "matrix_examples",
                                [sp.Matrix()]
                            )[0]
                        )
                    ),
                    "combine",
                    {
                        "transform":
                            node.get(
                                "transform",
                                None
                            ),

                        "transform_examples":
                            node.get(
                                "transform_examples",
                                None
                            )
                    }
                ]]

                for child_path in combination:

                    combined_path.extend(child_path)

                all_paths.append(combined_path)

            return all_paths

        return recurse(tree)




class PathEvaluator:



    @staticmethod
    def eval_symbolic(expr, n_value):

        if isinstance(expr, (int, float)):
            return float(expr)

        n = sp.Symbol('n')

        return float(
            sp.sympify(expr).subs(n, n_value)
        )

  
    @staticmethod
    def evaluate_path_beta(
        path,
        n_value,
        q,
        beta_func,
        verbose=True
    ):

        betas = []

        has_complex = False

        for idx, step in enumerate(path):

            dim, variance, typ, role, meta = step

            if role != "solve":
                continue

            dim_val = int(
                PathEvaluator.eval_symbolic(
                    dim,
                    n_value
                )
            )

            var_val = (
                PathEvaluator.eval_symbolic(
                    variance,
                    n_value
                )
            )



            if typ == "complex":

                dim_val *= 2
                var_val = var_val/2.

                has_complex = True

            if verbose:

                print(
                    f"testing for:"
                    f" {dim_val}, {q}, {var_val}"
                )

    

            Xs = estimator.ND.DiscreteGaussian(
                sqrt(var_val)
            )

            Xe = estimator.ND.DiscreteGaussian(
                sqrt(var_val)
            )



            l = beta_func[0](
                n=dim_val,
                q=q,
                Xs=Xs,
                Xe=Xe
            )
           
            try:
                result = beta_func[1](l)
            except Exception:
                result = {}
                result['beta'] = 2

            beta = result['beta']
            
            if verbose:
                print("beta: ", beta)

            betas.append((beta, idx))

        if not betas:
            return None

        max_beta, critical_idx = max(
            betas,
            key=lambda x: x[0]
        )

        return {
            "path": path,
            "beta": max_beta,
            "complex": has_complex,
            "critical_node": critical_idx
        }



    @staticmethod
    def best_beta_for_tree(
        paths,
        n_value,
        q,
        beta_func,
        verbose=True
    ):

        evaluated = []

        for path in paths:

            res = (
                PathEvaluator
                .evaluate_path_beta(
                    path,
                    n_value,
                    q,
                    beta_func,
                    verbose
                )
            )

            if res is not None:

                evaluated.append(res)

        if not evaluated:
            return None



        best_overall = min(
            evaluated,
            key=lambda x: (
                x["beta"],
                -int(x["complex"])
            )
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


# Main Security Estimator

class SecurityEstimator:

    def __init__(
        self,
        n,
        q,
        base_var,
        mul_function,
        level=1,
        lattice="NTRU",
        mul_type="LL",
        filename="examples.py",
        variable='h',
        parameters=None,
        symbolic = True  ######## default is symbolic = True for this level decomposer
    ):

        self.n = n
        self.q = q

        self.base_var = base_var

        self.mul_function = mul_function

        self.level = level
        self.lattice = lattice

        self.mul_type = mul_type

        self.filename = filename
        self.variable = variable
        self.parameters = parameters
        self.compilerobj = None
        self.symbolic = symbolic



    def construct_symbolic_matrix(self):
     

        # return compiler.construct_symbolic_matrix(
        #     filename=self.filename,
        #     funcname=self.mul_function,
        #     dimension=self.n,
        #     parameters=self.parameters,
        #     mul_type=self.mul_type,
        #     variable=self.variable
        # )

        compilerobj = AlgebraCompiler(
            filename   = self.filename,
            funcname   = self.mul_function,
            dimension  = self.n,
            parameters = self.parameters
        )

        compilerobj.extract_structure_tensor()
        self.compilerobj = compilerobj
        return compilerobj.return_symbolic_matrix(
            self.mul_type,
            self.variable
        )


   
    # Select decomposer
    def get_decomposer(self, H):

        match self.level:

            case 1:
                 
                return fastdec.SymbolicDecomposer(
                    symbolic_matrix=H,
                    n=self.n,
                    q=self.q,
                    base_var=self.base_var
                )
                
    
            case 2:

                return deepdec.PrimaryDecomposer(
                    symbolic_matrix=H,
                    n=self.n,
                    q=self.q,
                    base_var=self.base_var,
                    compilerobj = self.compiler.obj,  ##compilerobj and mul_type are used in getting basis and commutant easily 
                                                      ## in the non-symbolic form
                    mul_type = self.mul_type
                )

            case 3:

                return deepdec.ExhuastivDecomposer(
                    symbolic_matrix=H,
                    n=self.n,
                    q=self.q,
                    base_var=self.base_var,
                    compilerobj = self.compilerobj
                )

            case _:

                raise ValueError(
                    "Invalid decomposition level."
                )

  
    # Select lattice estimator
    def get_beta_function(self):

        if self.lattice == "NTRU":

            return (
                estimator.NTRU.Parameters,
                estimator.NTRU.primal_usvp
            )

        elif self.lattice == "LWE":

            return (
                estimator.LWE.Parameters,
                estimator.LWE.primal_usvp
            )

        raise ValueError(
            f"Unknown lattice type '{self.lattice}'"
        )

    
    # Global optimization
 
    def best_global_beta(
        self,
        generalized_trees,
        beta_func,
        verbose=True
    ):

        global_best = None

        global_best_integer = None

        for tree_idx, tree in enumerate(
            generalized_trees
        ):

            paths = (
                PathExtractor
                .get_all_paths_generalized(
                    tree
                )
            )

            result = (
                PathEvaluator
                .best_beta_for_tree(
                    paths=paths,
                    n_value=self.n,
                    q=self.q,
                    beta_func=beta_func,
                    verbose=verbose
                )
            )

            if result is None:
                continue

            
            # Best overall
      
            cand = result["best_overall"]

            if (
                global_best is None
                or (
                    cand["beta"]
                    < global_best["beta"]
                )
                or (
                    cand["beta"]
                    == global_best["beta"]
                    and cand["complex"]
                    and not global_best["complex"]
                )
            ):

                global_best = {
                    **cand,
                    "tree_index": tree_idx
                }

            
            # Best integer only
       
            if result["best_integer"] is not None:

                cand_int = result["best_integer"]

                if (
                    global_best_integer is None
                    or (
                        cand_int["beta"]
                        < global_best_integer["beta"]
                    )
                ):

                    global_best_integer = {
                        **cand_int,
                        "tree_index": tree_idx
                    }

        return {
            "best_global": global_best,
            "best_global_integer": global_best_integer,
            "concrete_beta":
                global_best_integer['beta']
        }

    
    # Main estimation 
    def estimate_security(
        self,
        verbose=True
    ):

        H = self.construct_symbolic_matrix()

        decomposer = self.get_decomposer(H)

        full_trees = (
            decomposer.get_full_trees(
                verbose=verbose
            )
        )

        beta_func = self.get_beta_function()

        result = self.best_global_beta(
            full_trees,
            beta_func,
            verbose
        )

        return result
