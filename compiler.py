import ast
import re
import sympy as sp


class AlgebraCompiler:
    """
    CoNAN Algebra Compiler
    ======================

    Recover the structure tensor of an associative algebra directly
    from a white-box implementation of a bilinear multiplication rule.

    The multiplication function must satisfy the restricted form

        h[...] += c * f[...] * g[...]

    where:
        • f and g are read-only input vectors,
        • h is a write-only output vector,
        • c is a scalar coefficient.

    Under these conditions, the compiler automatically recovers
    the structure tensor P satisfying

        e_i e_j = sum_k P[i][j][k] e_k

    with respect to the canonical basis.

    The compiler supports arbitrary additional algebra parameters.

    Example:
        mul(f, g, h, n, m, p)

    by passing:

        parameters = {
            "n": ...,
            "m": ...,
            "p": ...
        }
    """

    # Initialization
    def __init__(
        self,
        filename,
        funcname,
        dimension,
        parameters=None
    ):

        self.filename = filename
        self.funcname = funcname
        self.dimension = dimension



        if parameters is None:

            parameters = {
                "n": dimension
            }

        self.parameters = parameters

        self.source_code = None
        self.modified_code = None

        self.structure_tensor = None



    def load_source(self):

        with open(self.filename, "r") as f:
            self.source_code = f.read()

    def extract_function_ast(self):

        if self.source_code is None:
            self.load_source()

        tree = ast.parse(self.source_code)

        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == self.funcname
        ]

        if len(nodes) != 1:

            raise ValueError(
                f"Function '{self.funcname}' "
                f"not found uniquely."
            )

        return nodes[0]



    def generate_modified_function(self):
        """
        Rewrite:

            h[k] += c*f[i]*g[j]

        into:

            h[i][j][k] += c*f[i]*g[j]

        so that a single execution recovers the full structure tensor.
        """

        node = self.extract_function_ast()

        source_code = ast.unparse(node)

        lines = source_code.strip().split('\n')

     

        parameter_string = ", ".join(
            self.parameters.keys()
        )

        if parameter_string != "":
            parameter_string = ", " + parameter_string

   

        newlines = [
            f'def {self.funcname}_modified('
            f'f, g, h{parameter_string}):'
        ]

      
        # Bilinear update pattern
        pattern = re.compile(
            r'([ \t]*)(h)([ \t]*)(\[.*?\])([ \t]*)'
            r'(\+=|-=|=|:=)([ \t]*-?[ \t]*'
            r'((?:\d+|[a-zA-Z_]\w*)[ \t]*\*)?[ \t]*)'
            r'(f)([ \t]*)(\[.*?\])([ \t]*)'
            r'(\*)([ \t]*)(g)([ \t]*)(\[.*?\])'
        )

      

        for line in lines[1:]:

            match = pattern.fullmatch(line)

            if match is None:

                newlines.append(line)

                continue

            d = {
                i: match.group(i)
                for i in range(1, 18)
            }

            ###############################################################
            # Transform:
            #
            #     h[k] += c*f[i]*g[j]
            #
            # into:
            #
            #     h[i][j][k] += c*f[i]*g[j]
            ################################################################

            newline = (
                f'{d[1]}{d[2]}{d[3]}'
                f'{d[11]}{d[17]}'
                f'{d[4]}{d[5]}{d[6]}'
                f'{d[7]}{d[9]}{d[10]}'
                f'{d[11]}{d[12]}{d[13]}'
                f'{d[14]}{d[15]}{d[16]}{d[17]}'
            )

            newlines.append(newline)

        self.modified_code = '\n'.join(newlines)



    def extract_structure_tensor(self):

        self.load_source()

        self.generate_modified_function()

        namespace = {}

        exec(
            compile(
                f'{self.source_code}\n\n{self.modified_code}',
                "<generated>",
                "exec"
            ),
            namespace
        )

        mul_extracted = namespace[
            f'{self.funcname}_modified'
        ]

       

        f = [1] * self.dimension
        g = [1] * self.dimension

        

        P = [
            [
                [0 for _ in range(self.dimension)]
                for _ in range(self.dimension)
            ]
            for _ in range(self.dimension)
        ]

 

        mul_extracted(
            f,
            g,
            P,
            **self.parameters
        )

        self.structure_tensor = P

        return P

    
    # Associativity Verification

    def verify_associativity(self):

        if self.structure_tensor is None:

            raise ValueError(
                "Structure tensor has not been extracted."
            )

        P = self.structure_tensor

        n = self.dimension

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for s in range(n):

                        lhs = sum(
                            P[i][j][r] * P[r][k][s]
                            for r in range(n)
                        )

                        rhs = sum(
                            P[j][k][r] * P[i][r][s]
                            for r in range(n)
                        )

                        if lhs != rhs:
                            return False

        return True

   

    def return_lattice(self, mul_type='LL'):

        if self.structure_tensor is None:
            self.extract_structure_tensor()

        P = self.structure_tensor

        n = self.dimension

        if mul_type == 'LL':

            return [
                sp.Matrix([
                    [P[i][k][j] for j in range(n)]
                    for i in range(n)
                ])
                for k in range(n)
            ]

        if mul_type == 'LR':

            return [
                sp.Matrix([
                    [P[j][k][i] for j in range(n)]
                    for i in range(n)
                ])
                for k in range(n)
            ]

        if mul_type == 'RL':

            return [
                sp.Matrix([
                    [P[k][i][j] for j in range(n)]
                    for i in range(n)
                ])
                for k in range(n)
            ]

        if mul_type == 'RR':

            return [
                sp.Matrix([
                    [P[k][j][i] for j in range(n)]
                    for i in range(n)
                ])
                for k in range(n)
            ]

        return None


    def return_symbolic_matrix(
        self,
        mul_type='LL',
        variable='h'
    ):

        L = self.return_lattice(mul_type)

        h = sp.Matrix(
            sp.symbols(
                f'{variable}0:{self.dimension}'
            )
        )

        H = sp.zeros(self.dimension)

        for k in range(self.dimension):

            H += h[k] * L[k]

        return H


    def print_symbolic_matrix(
        self,
        mul_type='LL',
        variable='h'
    ):

        H = self.return_symbolic_matrix(
            mul_type,
            variable
        )

        print('\nSymbolic representation matrix:\n')

        sp.pprint(H)



    def compile(self):

        return {
            "dimension": self.dimension,
            "parameters": self.parameters,
            "function": self.funcname,
            "tensor": self.extract_structure_tensor(),
            "associative": self.verify_associativity()
        }



#  API

def extract_structure_tensor(
    filename,
    funcname,
    dimension,
    parameters=None
):

    compiler = AlgebraCompiler(
        filename=filename,
        funcname=funcname,
        dimension=dimension,
        parameters=parameters
    )

    return compiler.extract_structure_tensor()


def construct_lattice(
    filename,
    funcname,
    dimension,
    parameters=None,
    mul_type='LL'
):

    compiler = AlgebraCompiler(
        filename=filename,
        funcname=funcname,
        dimension=dimension,
        parameters=parameters
    )

    compiler.extract_structure_tensor()

    return compiler.return_lattice(mul_type)


def construct_symbolic_matrix(
    filename,
    funcname,
    dimension,
    parameters=None,
    mul_type='LL',
    variable='h'
):

    compiler = AlgebraCompiler(
        filename=filename,
        funcname=funcname,
        dimension=dimension,
        parameters=parameters
    )

    compiler.extract_structure_tensor()

    return compiler.return_symbolic_matrix(
        mul_type,
        variable
    )
    
    
###### call this function to build the commutant.    
def return_lattice(
    filename,
    funcname,
    dimension,
    parameters=None,
    mul_type='LL',
    variable='h'
):

    compiler = AlgebraCompiler(
        filename=filename,
        funcname=funcname,
        dimension=dimension,
        parameters=parameters
    )

    compiler.extract_structure_tensor()

    return compiler.return_lattice(mul_type)
    
