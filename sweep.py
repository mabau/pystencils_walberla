import sympy as sp
from collections import namedtuple
from jinja2 import Environment, PackageLoader

from pystencils import Field
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from pystencils.sympyextensions import assignments_from_python_function

KernelInfo = namedtuple("KernelInfo", ['ast', 'temporaryFields', 'fieldSwaps'])


class Sweep:
    def __init__(self, dim=3, f_size=None):
        self.dim = dim
        self.fSize = f_size
        self._fieldSwaps = []
        self._temporaryFields = []

    @staticmethod
    def constant(name):
        """Create a symbolic constant that is passed to the sweep as a parameter"""
        return sp.Symbol(name)

    def field(self, name, f_size=None):
        """Create a symbolic field that is passed to the sweep as BlockDataID"""
        # layout does not matter, since it is only used to determine order of spatial loops i.e. zyx, which is
        # always the same in waLBerla
        if self.dim is None:
            raise ValueError("Set the dimension of the sweep first, e.g. sweep.dim=3")
        return Field.create_generic(name, spatial_dimensions=self.dim, index_dimensions=1 if f_size else 0,
                                    layout='fzyx', index_shape=(f_size,) if f_size else None)

    def temporary_field(self, field, tmp_field_name=None):
        """Creates a temporary field as clone of field, which is swapped at the end of the sweep"""
        if tmp_field_name is None:
            tmp_field_name = field.name + "_tmp"
        self._temporaryFields.append(tmp_field_name)
        self._fieldSwaps.append((tmp_field_name, field.name))
        return Field.create_generic(tmp_field_name, spatial_dimensions=field.spatial_dimensions,
                                    index_dimensions=field.index_dimensions, layout=field.layout,
                                    index_shape=field.index_shape)

    @staticmethod
    def generate(name, sweep_function, namespace='pystencils', target='cpu',
                 dim=None, f_size=None, openmp=True):
        from pystencils_walberla.cmake_integration import codegen
        sweep = Sweep(dim, f_size)

        def generate_header_and_source():
            eqs = assignments_from_python_function(sweep_function, sweep=sweep)
            if target == 'cpu':
                from pystencils.cpu import create_kernel, add_openmp
                ast = create_kernel(eqs, function_name=name)
                if openmp:
                    add_openmp(ast, num_threads=openmp)
            elif target == 'gpu':
                from pystencils.gpucuda.kernelcreation import create_cuda_kernel
                ast = create_cuda_kernel(eqs, function_name=name)

            env = Environment(loader=PackageLoader('pystencils_walberla'))
            add_pystencils_filters_to_jinja_env(env)

            context = {
                'kernel': KernelInfo(ast, sweep._temporaryFields, sweep._fieldSwaps),
                'namespace': namespace,
                'className': ast.function_name[0].upper() + ast.function_name[1:],
                'target': target,
            }

            header = env.get_template("Sweep.tmpl.h").render(**context)
            source = env.get_template("Sweep.tmpl.cpp").render(**context)
            return header, source

        file_names = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(file_names, generate_header_and_source)

    @staticmethod
    def generate_from_equations(name, function_returning_equations, temporary_fields=[], field_swaps=[],
                                namespace="pystencils", target='cpu', openmp=True, **kwargs):
        from pystencils_walberla.cmake_integration import codegen

        def generate_header_and_source():
            eqs = function_returning_equations(**kwargs)

            if target == 'cpu':
                from pystencils.cpu import create_kernel, add_openmp
                ast = create_kernel(eqs, function_name=name)
                if openmp:
                    add_openmp(ast, num_threads=openmp)
            elif target == 'gpu':
                from pystencils.gpucuda.kernelcreation import create_cuda_kernel
                ast = create_cuda_kernel(eqs, function_name=name)

            env = Environment(loader=PackageLoader('pystencils_walberla'))
            add_pystencils_filters_to_jinja_env(env)

            context = {
                'kernel': KernelInfo(ast, temporary_fields, field_swaps),
                'namespace': namespace,
                'className': ast.function_name[0].upper() + ast.function_name[1:],
                'target': target,
            }

            header = env.get_template("Sweep.tmpl.h").render(**context)
            source = env.get_template("Sweep.tmpl.cpp").render(**context)
            return header, source

        file_names = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(file_names, generate_header_and_source)
