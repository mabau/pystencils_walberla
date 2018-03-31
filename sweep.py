import sympy as sp
from collections import namedtuple
from jinja2 import Environment, PackageLoader

from pystencils import Field
from pystencils_walberla.jinja_filters import addPystencilsFiltersToJinjaEnv
from pystencils.sympyextensions import assignments_from_python_function

KernelInfo = namedtuple("KernelInfo", ['ast', 'temporaryFields', 'fieldSwaps'])


class Sweep:
    def __init__(self, dim=3, fSize=None):
        self.dim = dim
        self.fSize = fSize
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
        if self.dim is None:
            raise ValueError("Set the dimension of the sweep first, e.g. sweep.dim=3")
        return Field.createGeneric(name, spatialDimensions=self.dim, indexDimensions=1 if fSize else 0,
                                   layout='fzyx', indexShape=(fSize,) if fSize else None)

    def temporaryField(self, field, tmpFieldName=None):
        """Creates a temporary field as clone of field, which is swapped at the end of the sweep"""
        if tmpFieldName is None:
            tmpFieldName = field.name + "_tmp"
        self._temporaryFields.append(tmpFieldName)
        self._fieldSwaps.append((tmpFieldName, field.name))
        return Field.createGeneric(tmpFieldName, spatialDimensions=field.spatialDimensions,
                                   indexDimensions=field.indexDimensions, layout=field.layout,
                                   indexShape=field.indexShape)

    @staticmethod
    def generate(name, sweep_function, namespace='pystencils', target='cpu',
                 dim=None, fSize=None, openMP=True):
        from pystencils_walberla.cmake_integration import codegen
        sweep = Sweep(dim, fSize)

        def generateHeaderAndSource():
            eqs = assignments_from_python_function(sweep_function, sweep=sweep)
            if target == 'cpu':
                from pystencils.cpu import createKernel, addOpenMP
                ast = createKernel(eqs, functionName=name)
                if openMP:
                    addOpenMP(ast, numThreads=openMP)
            elif target == 'gpu':
                from pystencils.gpucuda.kernelcreation import createCUDAKernel
                ast = createCUDAKernel(eqs, functionName=name)

            env = Environment(loader=PackageLoader('pystencils_walberla'))
            addPystencilsFiltersToJinjaEnv(env)

            context = {
                'kernel': KernelInfo(ast, sweep._temporaryFields, sweep._fieldSwaps),
                'namespace': namespace,
                'className': ast.functionName[0].upper() + ast.functionName[1:],
                'target': target,
            }

            header = env.get_template("Sweep.tmpl.h").render(**context)
            source = env.get_template("Sweep.tmpl.cpp").render(**context)
            return header, source

        fileNames = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(fileNames, generateHeaderAndSource)

    @staticmethod
    def generateFromEquations(name, function_returning_equations, temporaryFields=[], fieldSwaps=[],
                              namespace="pystencils", target='cpu', openMP=True, **kwargs):
        from pystencils_walberla.cmake_integration import codegen

        def generateHeaderAndSource():
            eqs = function_returning_equations(**kwargs)

            if target == 'cpu':
                from pystencils.cpu import createKernel, addOpenMP
                ast = createKernel(eqs, functionName=name)
                if openMP:
                    addOpenMP(ast, numThreads=openMP)
            elif target == 'gpu':
                from pystencils.gpucuda.kernelcreation import createCUDAKernel
                ast = createCUDAKernel(eqs, functionName=name)

            env = Environment(loader=PackageLoader('pystencils_walberla'))
            addPystencilsFiltersToJinjaEnv(env)

            context = {
                'kernel': KernelInfo(ast, temporaryFields, fieldSwaps),
                'namespace': namespace,
                'className': ast.functionName[0].upper() + ast.functionName[1:],
                'target': target,
            }

            header = env.get_template("Sweep.tmpl.h").render(**context)
            source = env.get_template("Sweep.tmpl.cpp").render(**context)
            return header, source

        fileNames = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(fileNames, generateHeaderAndSource)
