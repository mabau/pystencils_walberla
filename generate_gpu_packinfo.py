from pystencils import Field, FieldType, Assignment, create_kernel
from jinja2 import Environment, PackageLoader
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from collections import namedtuple

KernelInfo = namedtuple("KernelInfo", ['ast', 'temporary_fields', 'field_swaps'])


def generate_pack_info(class_name, directions_to_pack_terms, **create_kernel_params):
    target = create_kernel_params.get('target', 'cpu')

    fields_accessed = set()
    for terms in directions_to_pack_terms.values():
        for term in terms:
            assert isinstance(term, Field.Access)
            fields_accessed.add(term)

    data_types = {fa.field.dtype for fa in fields_accessed}
    if len(data_types) != 1:
        raise NotImplementedError("Fields of different data types are used - this is not supported")
    dtype = data_types.pop()

    buffer = Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER, dtype=dtype.numpy_dtype)

    kernels = {}
    all_accesses = set()
    elements_per_cell = {}
    for direction_set, terms in directions_to_pack_terms.items():
        all_accesses.update(terms)
        pack_ast = create_kernel([Assignment(buffer.center, term) for term in terms], **create_kernel_params)
        pack_ast.function_name = 'pack_{}'.format("".join(direction_set))
        unpack_ast = create_kernel([Assignment(term, buffer.center) for term in terms], **create_kernel_params)
        unpack_ast.function_name = 'unpack_{}'.format("".join(direction_set))
        kernels[direction_set] = (KernelInfo(pack_ast, [], []),
                                  KernelInfo(unpack_ast, [], []))
        elements_per_cell[direction_set] = len(terms)

    fused_kernel = create_kernel([Assignment(buffer.center, t) for t in all_accesses], **create_kernel_params)

    context = {
        'class_name': class_name,
        'kernels': kernels,
        'fused_kernel': KernelInfo(fused_kernel, [], []),
        'elements_per_cell': elements_per_cell,
        'target': target,
        'dtype': dtype,
    }

    env = Environment(loader=PackageLoader('pystencils_walberla'))
    add_pystencils_filters_to_jinja_env(env)
    header = env.get_template("GpuPackInfo.tmpl.h").render(**context)
    source = env.get_template("GpuPackInfo.tmpl.cpp").render(**context)
    return header, source


if __name__ == '__main__':
    f = Field.create_generic('f', spatial_dimensions=3, index_dimensions=0, layout='fzyx')
    spec = {('E', 'W', 'S', 'N'): [f.center]}
    header, source = generate_pack_info('GenGpuPackInfo', spec, target='gpu')
    print(header, file=open('/local/bauer/code/walberla/tests/cuda/GenGpuPackInfo.h', 'w'))
    print(source, file=open('/local/bauer/code/walberla/tests/cuda/GenGpuPackInfo.cu', 'w'))
    print("Done")
