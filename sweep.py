import sympy as sp
from collections import namedtuple
import functools
from jinja2 import Environment, PackageLoader

from pystencils import kernel as kernel_decorator, create_staggered_kernel
from pystencils import Field, SymbolCreator, create_kernel
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env

KernelInfo = namedtuple("KernelInfo", ['ast', 'temporary_fields', 'field_swaps', 'varying_parameters'])


class Sweep:
    const = SymbolCreator()

    def __init__(self, dim=3, f_size=None):
        self.dim = dim
        self.f_size = f_size
        self._field_swaps = []
        self._temporary_fields = []

    @staticmethod
    def constant(name):
        """Create a symbolic constant that is passed to the sweep as a parameter"""
        return sp.Symbol(name)

    def field(self, name, f_size=None):
        """Create a symbolic field that is passed to the sweep as BlockDataID"""
        # layout does not matter, since it is only used to determine order of spatial loops i.e. zyx, which is
        # always the same in walberla
        if self.dim is None:
            raise ValueError("Set the dimension of the sweep first, e.g. sweep.dim=3")
        return Field.create_generic(name, spatial_dimensions=self.dim, index_dimensions=1 if f_size else 0,
                                    layout='fzyx', index_shape=(f_size,) if f_size else None)

    def temporary_field(self, field, tmp_field_name=None):
        """Creates a temporary field as clone of field, which is swapped at the end of the sweep"""
        if tmp_field_name is None:
            tmp_field_name = field.name + "_tmp"
        self._temporary_fields.append(tmp_field_name)
        self._field_swaps.append((tmp_field_name, field.name))
        return Field.create_generic(tmp_field_name, spatial_dimensions=field.spatial_dimensions,
                                    index_dimensions=field.index_dimensions, layout=field.layout,
                                    index_shape=field.index_shape)

    @staticmethod
    def generate(name, sweep_function, namespace='pystencils', target='cpu',
                 dim=None, f_size=None, optimization={},):
        from pystencils_walberla.cmake_integration import codegen
        sweep = Sweep(dim, f_size)

        func = functools.partial(kernel_decorator, sweep_function, sweep=sweep)
        cb = functools.partial(Sweep._generate_header_and_source, func, name, target,
                               namespace, sweep._temporary_fields, sweep._field_swaps, optimization=optimization)

        file_names = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(file_names, cb)

    @staticmethod
    def generate_from_equations(name, function_returning_assignments, temporary_fields=[], field_swaps=[],
                                namespace="pystencils", target='cpu', optimization={},
                                staggered=False, varying_parameters=[], **kwargs):
        from pystencils_walberla.cmake_integration import codegen
        cb = functools.partial(Sweep._generate_header_and_source, function_returning_assignments, name, target,
                               namespace, temporary_fields, field_swaps,
                               optimization=optimization, staggered=staggered,
                               varying_parameters=varying_parameters, **kwargs)
        file_names = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(file_names, cb)

    @staticmethod
    def generate_pack_info(name, function_returning_assignments, target='gpu', **kwargs):
        from pystencils_walberla.cmake_integration import codegen

        def callback():
            from pystencils_walberla.generate_packinfo import generate_pack_info_from_kernel
            assignments = function_returning_assignments()
            return generate_pack_info_from_kernel(name, assignments, target=target, **kwargs)

        file_names = [name + ".h", name + ('.cpp' if target == 'cpu' else '.cu')]
        codegen.register(file_names, callback)

    @staticmethod
    def _generate_header_and_source(function_returning_equations, name, target, namespace,
                                    temporary_fields, field_swaps, optimization, staggered,
                                    varying_parameters, **kwargs):
        eqs = function_returning_equations(**kwargs)

        if not staggered:
            ast = create_kernel(eqs, target=target, **optimization)
        else:
            ast = create_staggered_kernel(*eqs, target=target, **optimization)
        ast.function_name = name

        env = Environment(loader=PackageLoader('pystencils_walberla'))
        add_pystencils_filters_to_jinja_env(env)

        context = {
            'kernel': KernelInfo(ast, temporary_fields, field_swaps, varying_parameters),
            'namespace': namespace,
            'class_name': ast.function_name[0].upper() + ast.function_name[1:],
            'target': target,
        }

        header = env.get_template("Sweep.tmpl.h").render(**context)
        source = env.get_template("Sweep.tmpl.cpp").render(**context)
        return header, source
