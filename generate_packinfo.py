from itertools import product
from collections import defaultdict, OrderedDict
from typing import Dict, Sequence, Tuple, Optional
from jinja2 import Environment, PackageLoader
from pystencils import Field, FieldType, Assignment, create_kernel
from pystencils.stencils import offset_to_direction_string, inverse_direction
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from pystencils_walberla.sweep import KernelInfo


def comm_directions(direction):
    direction = inverse_direction(direction)
    yield direction
    for i in range(len(direction)):
        if direction[i] != 0:
            dir_as_list = list(direction)
            dir_as_list[i] = 0
            if not all(e == 0 for e in dir_as_list):
                yield tuple(dir_as_list)


def generate_pack_info_for_field(class_name: str, field: Field,
                                 direction_subset: Optional[Tuple[Tuple[int, int, int]]] = None,
                                 **create_kernel_params):
    if not direction_subset:
        direction_subset = tuple((i, j, k) for i, j, k in product(*[(-1, 0, 1)] * 3))

    all_index_accesses = [field(*ind) for ind in product(*[range(s) for s in field.index_shape])]
    return generate_pack_info(class_name, {direction_subset: all_index_accesses}, **create_kernel_params)


def generate_pack_info_from_kernel(class_name: str, assignments: Sequence[Assignment], **create_kernel_params):
    reads = set()
    for a in assignments:
        reads.update(a.rhs.atoms(Field.Access))
    spec = defaultdict(set)
    for fa in reads:
        assert all(abs(e) <= 1 for e in fa.offsets)
        for comm_dir in comm_directions(fa.offsets):
            spec[(comm_dir,)].add(fa.field.center(*fa.index))
    return generate_pack_info(class_name, spec, **create_kernel_params)


def generate_pack_info(class_name: str,
                       directions_to_pack_terms: Dict[Tuple[Tuple], Sequence[Field.Access]],
                       namespace='pystencils',
                       **create_kernel_params):

    items = [(e[0], sorted(e[1], key=lambda x: str(x))) for e in directions_to_pack_terms.items()]
    items = sorted(items, key=lambda e: e[0])
    directions_to_pack_terms = OrderedDict(items)
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

    context = {
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
    header = env.get_template("GpuPackInfo.tmpl.h").render(**context)
    source = env.get_template("GpuPackInfo.tmpl.cpp").render(**context)
    return header, source
