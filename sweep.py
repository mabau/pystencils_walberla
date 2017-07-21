import sympy as sp
import inspect
import os
from collections import namedtuple
from jinja2 import Environment, PackageLoader

from pystencils import Field
from pystencils.cpu import createKernel
from pystencils.gpucuda.kernelcreation import createCUDAKernel
from pystencils_walberla.jinja_filters import addPystencilsFiltersToJinjaEnv


class Sweep:
    """
    Class to generate a waLBerla sweep from a pystencils kernel
    
    Example to generate a simple waLBerla sweep source-destination sweep that multiplies every
    cell by a constant 'h'. 
     
    >>> k = Sweep(dim=3)
    >>> src = k.field("f1")
    >>> dst = k.temporaryField(src)
    >>> h = k.constant("h")
    >>> k.addEq(dst[0,0,0], src[0,0,0] * h)
    >>> #k.generate()
    
    For a real scenario the generate function has to be called (this is just skipped in this doctest)
    The snippet above should be located in a separate Python file that has one of the following endings
     - .gen.py       to generate CPU code
     - .cuda.gen.py  to generate CUDA code
    
    It then generates a two files with the same name but endings .cpp (.cu) and .h
    """
    def __init__(self, dim=3, fSize=None):
        self.dim = dim
        self.fSize = fSize
        self.eqs = []
        self._fieldSwaps = []
        self._temporaryFields = []

    @staticmethod
    def constant(name):
        """Create a symbolic constant that is passed to the sweep as a parameter"""
        return sp.Symbol(name)

    def field(self, name, fSize=None):
        """Create a symbolic field that is passed to the sweep as BlockDataID"""
        # layout does not matter, since it is only used to determine order of spatial loops i.e. zyx, which is
        # always the same in waLBerla
        return Field.createGeneric(name, spatialDimensions=self.dim, indexDimensions=1 if fSize else 0, layout='fzyx')

    def temporaryField(self, field):
        """Creates a temporary field as clone of field, which is swapped at the end of the sweep"""
        tmpFieldName = field.name + "_tmp"
        self._temporaryFields.append(tmpFieldName)
        self._fieldSwaps.append((tmpFieldName, field.name))
        return Field.createGeneric(tmpFieldName, spatialDimensions=field.spatialDimensions,
                                   indexDimensions=field.indexDimensions, layout=field.layout)

    def addEq(self, lhs, rhs):
        """Add an update equation to the pystencils kernel"""
        self.eqs.append(sp.Eq(lhs, rhs))

    def generate(self, namespace="pystencils"):
        """Call this function at the end to generate the corresponding .cpp(.cu) and .h files"""
        scriptFileName = inspect.stack()[-1][1]
        if scriptFileName.endswith(".cuda.gen.py"):
            fileName = scriptFileName[:-len(".cuda.gen.py")]
            fileName = os.path.split(fileName)[1]
            target = 'gpu'
        elif scriptFileName.endswith(".gen.py"):
            fileName = scriptFileName[:-len(".gen.py")]
            fileName = os.path.split(fileName)[1]
            target = 'cpu'
        else:
            fileName = "GeneratedKernel"
            target = 'cpu'

        if target == 'cpu':
            ast = createKernel(self.eqs, functionName=fileName)
        elif target == 'gpu':
            ast = createCUDAKernel(self.eqs, functionName=fileName)

        env = Environment(loader=PackageLoader('pystencils_walberla'))
        addPystencilsFiltersToJinjaEnv(env)

        KernelInfo = namedtuple("KernelInfo", ['ast', 'temporaryFields', 'fieldSwaps'])

        context = {
            'kernel': KernelInfo(ast, self._temporaryFields, self._fieldSwaps),
            'namespace': namespace,
            'className': ast.functionName[0].upper() + ast.functionName[1:],
            'target': target,
        }

        with open(fileName + ".h", 'w') as f:
            content = env.get_template("Sweep.tmpl.h").render(**context)
            f.write(content)

        suffix = '.cu' if target == 'gpu' else '.cpp'
        with open(fileName + suffix, 'w') as f:
            content = env.get_template("Sweep.tmpl.cpp").render(**context)
            f.write(content)


