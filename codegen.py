from jinja2 import Environment, PackageLoader
from collections import OrderedDict, defaultdict
from itertools import product
from typing import Dict, Sequence, Tuple, Optional

from pystencils import create_staggered_kernel, Field, create_kernel, Assignment, FieldType
from pystencils.backends.cbackend import get_headers
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
from pystencils.stencils import offset_to_direction_string, inverse_direction
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env


__all__ = ['generate_sweep', 'generate_pack_info', 'generate_pack_info_for_field', 'generate_pack_info_from_kernel',
           'default_create_kernel_parameters', 'KernelInfo']


def generate_sweep(generation_context, class_name, assignments,
                   namespace='pystencils', field_swaps=(), staggered=False, varying_parameters=(),
                   inner_outer_split=False,
                   **create_kernel_params):
    if hasattr(assignments, 'all_assignments'):
        assignments = assignments.all_assignments

    create_kernel_params = default_create_kernel_parameters(generation_context, create_kernel_params)

    if not staggered:
        ast = create_kernel(assignments, **create_kernel_params)
    else:
        ast = create_staggered_kernel(*assignments, **create_kernel_params)

    def to_name(f):
        return f.name if isinstance(f, Field) else f

    field_swaps = tuple((to_name(e[0]), to_name(e[1])) for e in field_swaps)
    temporary_fields = tuple(e[1] for e in field_swaps)

    ast.function_name = class_name.lower()

    env = Environment(loader=PackageLoader('pystencils_walberla'))
    add_pystencils_filters_to_jinja_env(env)

    if inner_outer_split is False:
        jinja_context = {
            'kernel': KernelInfo(ast, temporary_fields, field_swaps, varying_parameters),
            'namespace': namespace,
            'class_name': class_name,
            'target': create_kernel_params.get("target", "cpu"),
            'headers': get_headers(ast),
        }
        header = env.get_template("Sweep.tmpl.h").render(**jinja_context)
        source = env.get_template("Sweep.tmpl.cpp").render(**jinja_context)
    else:
        main_kernel_info = KernelInfo(ast, temporary_fields, field_swaps, varying_parameters)
        representative_field = {p.field_name for p in main_kernel_info.parameters if p.is_field_parameter}
        representative_field = sorted(representative_field)[0]

        jinja_context = {
            'kernel': main_kernel_info,
            'namespace': namespace,
            'class_name': class_name,
            'target': create_kernel_params.get("target", "cpu"),
            'field': representative_field,
            'headers': get_headers(ast),
        }
        header = env.get_template("SweepInnerOuter.tmpl.h").render(**jinja_context)
        source = env.get_template("SweepInnerOuter.tmpl.cpp").render(**jinja_context)

    source_extension = "cpp" if create_kernel_params.get("target", "cpu") == "cpu" else "cu"
    generation_context.write_file("{}.h".format(class_name), header)
    generation_context.write_file("{}.{}".format(class_name, source_extension), source)


def generate_pack_info_for_field(generation_context, class_name: str, field: Field,
                                 direction_subset: Optional[Tuple[Tuple[int, int, int]]] = None,
                                 **create_kernel_params):
    if not direction_subset:
        direction_subset = tuple((i, j, k) for i, j, k in product(*[(-1, 0, 1)] * 3))

    all_index_accesses = [field(*ind) for ind in product(*[range(s) for s in field.index_shape])]
    return generate_pack_info(generation_context, class_name, {direction_subset: all_index_accesses},
                              **create_kernel_params)


def generate_pack_info_from_kernel(generation_context, class_name: str, assignments: Sequence[Assignment],
                                   **create_kernel_params):
    reads = set()
    for a in assignments:
        reads.update(a.rhs.atoms(Field.Access))
    spec = defaultdict(set)
    for fa in reads:
        assert all(abs(e) <= 1 for e in fa.offsets)
        for comm_dir in comm_directions(fa.offsets):
            spec[(comm_dir,)].add(fa.field.center(*fa.index))
    return generate_pack_info(generation_context, class_name, spec, **create_kernel_params)


def generate_pack_info(generation_context, class_name: str,
                       directions_to_pack_terms: Dict[Tuple[Tuple], Sequence[Field.Access]],
                       namespace='pystencils',
                       **create_kernel_params):
    items = [(e[0], sorted(e[1], key=lambda x: str(x))) for e in directions_to_pack_terms.items()]
    items = sorted(items, key=lambda e: e[0])
    directions_to_pack_terms = OrderedDict(items)

    create_kernel_params = default_create_kernel_parameters(generation_context, create_kernel_params)
    target = create_kernel_params.get('target', 'cpu')

    fields_accessed = set()
    for terms in directions_to_pack_terms.values():
        for term in terms:
            assert isinstance(term, Field.Access) and all(e == 0 for e in term.offsets)
            fields_accessed.add(term)

    field_names = {fa.field.name for fa in fields_accessed}

    data_types = {fa.field.dtype for fa in fields_accessed}
    if len(data_types) != 1:
        raise NotImplementedError("Fields of different data types are used - this is not supported")
    dtype = data_types.pop()

    pack_kernels = OrderedDict()
    unpack_kernels = OrderedDict()
    all_accesses = set()
    elements_per_cell = OrderedDict()
    for direction_set, terms in directions_to_pack_terms.items():
        for d in direction_set:
            if not all(abs(i) <= 1 for i in d):
                raise NotImplementedError("Only first neighborhood supported")

        buffer = Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER,
                                      dtype=dtype.numpy_dtype, index_shape=(len(terms),))

        direction_strings = tuple(offset_to_direction_string(d) for d in direction_set)
        inv_direction_string = tuple(offset_to_direction_string(inverse_direction(d)) for d in direction_set)
        all_accesses.update(terms)

        pack_ast = create_kernel([Assignment(buffer(i), term) for i, term in enumerate(terms)],
                                 **create_kernel_params)
        pack_ast.function_name = 'pack_{}'.format("_".join(direction_strings))
        unpack_ast = create_kernel([Assignment(term, buffer(i)) for i, term in enumerate(terms)],
                                   **create_kernel_params)
        unpack_ast.function_name = 'unpack_{}'.format("_".join(inv_direction_string))

        pack_kernels[direction_strings] = KernelInfo(pack_ast)
        unpack_kernels[inv_direction_string] = KernelInfo(unpack_ast)
        elements_per_cell[direction_strings] = len(terms)

    fused_kernel = create_kernel([Assignment(buffer.center, t) for t in all_accesses], **create_kernel_params)

    jinja_context = {
        'class_name': class_name,
        'pack_kernels': pack_kernels,
        'unpack_kernels': unpack_kernels,
        'fused_kernel': KernelInfo(fused_kernel),
        'elements_per_cell': elements_per_cell,
        'target': target,
        'dtype': dtype,
        'field_name': field_names.pop(),
        'namespace': namespace,
    }

    env = Environment(loader=PackageLoader('pystencils_walberla'))
    add_pystencils_filters_to_jinja_env(env)
    header = env.get_template("GpuPackInfo.tmpl.h").render(**jinja_context)
    source = env.get_template("GpuPackInfo.tmpl.cpp").render(**jinja_context)

    source_extension = "cpp" if create_kernel_params.get("target", "cpu") == "cpu" else "cu"
    generation_context.write_file("{}.h".format(class_name), header)
    generation_context.write_file("{}.{}".format(class_name, source_extension), source)


# ---------------------------------- Internal --------------------------------------------------------------------------


class KernelInfo:
    def __init__(self, ast, temporary_fields=(), field_swaps=(), varying_parameters=()):
        self.ast = ast
        self.temporary_fields = tuple(temporary_fields)
        self.field_swaps = tuple(field_swaps)
        self.varying_parameters = tuple(varying_parameters)
        self.parameters = ast.get_parameters()  # cache parameters here


def default_create_kernel_parameters(generation_context, params):
    default_dtype = "float64" if generation_context.double_accuracy else 'float32'

    if generation_context.optimize_for_localhost:
        default_vec_is = get_supported_instruction_sets()[-1]
    else:
        default_vec_is = None

    params['target'] = params.get('target', 'cpu')
    params['data_type'] = params.get('data_type', default_dtype)
    params['cpu_openmp'] = params.get('cpu_openmp', generation_context.openmp)
    params['cpu_vectorize_info'] = params.get('cpu_vectorize_info', {})

    vec = params['cpu_vectorize_info']
    vec['instruction_set'] = vec.get('instruction_set', default_vec_is)
    vec['assume_inner_stride_one'] = True
    vec['assume_aligned'] = vec.get('assume_aligned', False)
    vec['nontemporal'] = vec.get('nontemporal', False)
    return params


def comm_directions(direction):
    direction = inverse_direction(direction)
    yield direction
    for i in range(len(direction)):
        if direction[i] != 0:
            dir_as_list = list(direction)
            dir_as_list[i] = 0
            if not all(e == 0 for e in dir_as_list):
                yield tuple(dir_as_list)
